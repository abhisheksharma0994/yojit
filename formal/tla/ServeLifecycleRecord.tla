---- MODULE ServeLifecycleRecord ----
\* Model of `server.py` and `cli.py` AFTER the ownership/record fix.  Same port,
\* same single-server design -- what changed is who is allowed to kill what, and
\* which writes share a critical section.
\*
\*   locking.state_lock()   an advisory lock (flock) serialising state writes
\*   _free_port(PORT)       kills the listener ONLY when server.json names it
\*                          (pid AND port match); otherwise it raises
\*                          PortOwnedByAnotherProcess and the launch is abandoned
\*   server.json            {pid, port, model, context, output, started_at}
\*   set_default + sync     one critical section, not two
\*   cmd_stop / cmd_status  read server.json and check pid_alive, instead of
\*                          trusting `lsof` ("whatever holds 8080 is ours")
\*
\* Contrast with ServeLifecycle.tla, which models the pre-fix code: there
\* _free_port killed whatever held the port, and set_default/sync could
\* interleave with a second process.  Its ReportedServerIsAlive and
\* DefaultMatchesAdvertised are both violated; here the same shape is fixed and
\* three invariants hold.
\*
\* FOREIGN is the listener yojit never started -- someone else's service on 8080.
\* It has no entry in server.json and no yojit process ever owned it.
EXTENDS Integers, TLC

CONSTANT N
Procs == 1..N
\* Deliberately outside Procs and outside 0, which is the "nothing here" sentinel
\* used by server/owner/record/statusVar -- otherwise "the port is held by
\* someone else" would be indistinguishable from "the port is free".
FOREIGN == -1

\* server:   the model whose server is listening on PORT (0 = port free, FOREIGN = not ours)
\* owner:    yojit process that started it (0 when none)
\* record:   what server.json names (0 = absent) -- the ONLY authority to kill
\* pc[i]:    start -> launching -> recorded -> committed -> done, or aborted
\* victims:  every listener _free_port has killed
\* statusVar: what `yojit status` last reported as "the server", 0 = nothing
\* lock:     0 = free, otherwise the process holding it
VARIABLES server, owner, record, pc, default, advertised, victims, statusVar, lock

vars == <<server, owner, record, pc, default, advertised, victims, statusVar, lock>>

Init ==
  /\ server = FOREIGN        \* port starts held by someone else's service
  /\ owner = 0
  /\ record = 0
  /\ pc = [i \in Procs |-> "start"]
  /\ default = 0
  /\ advertised = 0
  /\ victims = {}
  /\ statusVar = 0
  /\ lock = 0

\* _free_port + launch + health loop + _record_server, all inside the one
\* LAUNCH_LOCK_TIMEOUT critical section.  Reclaiming the port is only permitted
\* when server.json names the current listener -- that is the whole fix.  A
\* FOREIGN holder fails every disjunct, so only Refuse(i) is enabled for it.
BeginLaunch(i) ==
  /\ pc[i] = "start"
  /\ lock = 0
  /\ \/ server = 0                              \* port already free
     \/ (server = record /\ record # 0)         \* ours, per the record
  /\ lock' = i
  /\ victims' = IF server = 0 THEN victims ELSE victims \cup {server}
  /\ server' = 0
  /\ owner' = 0
  /\ record' = 0                                \* cleared with the kill
  /\ pc' = [pc EXCEPT ![i] = "launching"]
  /\ UNCHANGED <<default, advertised, statusVar>>

\* The record's own listener has died since it was written (OOM, crash, reboot).
\* The record stays stale on disk; only pid_alive can tell, which is exactly why
\* status/stop consult it.  Without this action the model would never explore the
\* stale-record states that the status fix exists for.
Crash ==
  /\ server # 0
  /\ server # FOREIGN
  /\ server' = 0
  /\ owner' = 0
  /\ UNCHANGED <<record, pc, default, advertised, victims, statusVar, lock>>

\* Port held by a process yojit did not start: refuse, kill nothing, launch nothing.
Refuse(i) ==
  /\ pc[i] = "start"
  /\ server = FOREIGN
  /\ pc' = [pc EXCEPT ![i] = "aborted"]
  /\ UNCHANGED <<server, owner, record, default, advertised, victims, statusVar, lock>>

Bind(i) ==
  /\ pc[i] = "launching"
  /\ lock = i
  /\ server' = i
  /\ owner' = i
  /\ UNCHANGED <<record, pc, default, advertised, victims, statusVar, lock>>

\* _record_server writes server.json and the launch lock is released here -- the
\* real code does not hold one lock across launch and publish, which is why
\* serve() re-validates before handing off to opencode.
Record(i) ==
  /\ pc[i] = "launching"
  /\ lock = i
  /\ server = i
  /\ record' = i
  /\ lock' = 0
  /\ pc' = [pc EXCEPT ![i] = "recorded"]
  /\ UNCHANGED <<server, owner, default, advertised, victims, statusVar>>

\* manifest.set_default(model_id) and opencode_sync.sync(...) share ONE critical
\* section.  In ServeLifecycle.tla they were two separate writes, and a second
\* process could land between them.
Commit(i) ==
  /\ pc[i] = "recorded"
  /\ lock = 0
  /\ lock' = i
  /\ default' = i
  /\ advertised' = i
  /\ pc' = [pc EXCEPT ![i] = "committed"]
  /\ UNCHANGED <<server, owner, record, victims, statusVar>>

Release(i) ==
  /\ pc[i] = "committed"
  /\ lock = i
  /\ statusVar' = IF record = i /\ server = i THEN i ELSE 0
  /\ lock' = 0
  /\ pc' = [pc EXCEPT ![i] = "done"]
  /\ UNCHANGED <<server, owner, record, default, advertised, victims>>

\* `yojit stop`: read server.json, kill only that pid, clear the record.
Stop ==
  /\ record # 0
  /\ server = record
  /\ victims' = victims \cup {record}
  /\ server' = 0
  /\ owner' = 0
  /\ record' = 0
  /\ UNCHANGED <<pc, default, advertised, statusVar, lock>>

\* `yojit status`: the record and pid_alive, never the bare port holder.
Status ==
  /\ statusVar' = IF record # 0 /\ server = record THEN record ELSE 0
  /\ UNCHANGED <<server, owner, record, pc, default, advertised, victims, lock>>

Next ==
  \/ \E i \in Procs :
       BeginLaunch(i) \/ Bind(i) \/ Record(i) \/ Commit(i) \/ Release(i) \/ Refuse(i)
  \/ Crash \/ Stop \/ Status

Spec == Init /\ [][Next]_vars

\* Someone else's service on 8080 is never killed, before or after a refusal.
ForeignListenerNeverKilled ==
  FOREIGN \notin victims

\* The public claim -- opencode.json's "(running)" marker -- and the manifest's
\* default slot are written together, so they can never disagree about which
\* model is live.  This is DefaultMatchesAdvertised from ServeLifecycle.tla,
\* which that spec violates.
DefaultMatchesAdvertised ==
  \A i \in Procs : pc[i] = "done" => default = advertised

\* status never reports a listener yojit did not start, and never reports a
\* crashed one: the pre-fix code reported whatever _port_pid returned.
StatusNeverReportsForeign ==
  statusVar # FOREIGN

StatusOnlyReportsLiveServers ==
  statusVar # 0 => server = statusVar

====
