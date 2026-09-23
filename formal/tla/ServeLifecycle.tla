---- MODULE ServeLifecycle ----
\* Model of `server.serve()` on one fixed port (PORT = 8080, the tool's design
\* is a single server on a single port).
\*
\*   _free_port(PORT)      kill whatever is listening -- no ownership check,
\*                         no cooperative lock with any other yojit invocation
\*   backend.launch(...)   bind the port
\*   health_check loop     wait up to 120s for the server to answer
\*   manifest.set_default  read-modify-write on the manifest
\*   opencode_sync.sync    read-modify-write on opencode.json, marking the model
\*                         "(running)"
\*   print "Server running in the background (PID ...)" then exec opencode
\*
\* Two invariants are checked, in two separate configs:
\*   ReportedServerIsAlive    a process that reported success still owns a live server
\*   DefaultMatchesAdvertised the manifest default agrees with what opencode.json
\*                            advertises as running
EXTENDS Integers, TLC

CONSTANT N
Procs == 1..N

\* server: model id listening on PORT, 0 when the port is free
\* owner:  yojit process that started it, 0 when the port is free
\* pc[i]:  start -> launched -> up -> defaulted -> synced -> reported -> done
\* default:    manifest's default model id (0 = unset)
\* advertised: model id marked "(running)" in opencode.json (0 = none)
VARIABLES server, owner, pc, default, advertised

vars == <<server, owner, pc, default, advertised>>

Init ==
  /\ server = 0
  /\ owner = 0
  /\ pc = [i \in Procs |-> "start"]
  /\ default = 0
  /\ advertised = 0

FreePort(i) ==
  /\ pc[i] = "start"
  /\ server' = 0
  /\ owner' = 0
  /\ pc' = [pc EXCEPT ![i] = "launched"]
  /\ UNCHANGED <<default, advertised>>

Launch(i) ==
  /\ pc[i] = "launched"
  /\ server' = i
  /\ owner' = i
  /\ pc' = [pc EXCEPT ![i] = "up"]
  /\ UNCHANGED <<default, advertised>>

HealthOk(i) ==
  /\ pc[i] = "up"
  /\ server = i                     \* the check the real code performs
  /\ pc' = [pc EXCEPT ![i] = "defaulted"]
  /\ UNCHANGED <<server, owner, default, advertised>>

SetDefault(i) ==
  /\ pc[i] = "defaulted"
  /\ default' = i
  /\ pc' = [pc EXCEPT ![i] = "synced"]
  /\ UNCHANGED <<server, owner, advertised>>

Sync(i) ==
  /\ pc[i] = "synced"
  /\ advertised' = i
  /\ pc' = [pc EXCEPT ![i] = "reported"]
  /\ UNCHANGED <<server, owner, default>>

Report(i) ==
  /\ pc[i] = "reported"
  /\ pc' = [pc EXCEPT ![i] = "done"]
  /\ UNCHANGED <<server, owner, default, advertised>>

Next == \E i \in Procs :
  FreePort(i) \/ Launch(i) \/ HealthOk(i) \/ SetDefault(i) \/ Sync(i) \/ Report(i)

Spec == Init /\ [][Next]_vars

\* A process that has reported success, and therefore handed off to opencode
\* bound to this model, must still have its server alive on PORT.
ReportedServerIsAlive ==
  \A i \in Procs : pc[i] = "done" => (server = i /\ owner = i)

\* The model in the manifest's default slot and the one opencode.json advertises
\* as running must agree, once a process has written both.
DefaultMatchesAdvertised ==
  \A i \in Procs : pc[i] \in {"reported", "done"} => (default = advertised)

====
