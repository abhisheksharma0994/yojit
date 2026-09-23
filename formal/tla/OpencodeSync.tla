---- MODULE OpencodeSync ----
\* Model of `opencode_sync.sync()`: a read-modify-write on
\* ~/.config/opencode/opencode.json with no lock.
\*
\*   config = json.loads(CONFIG_PATH.read_text())   \* read
\*   local_provider["models"] = models_obj          \* modify in memory
\*   CONFIG_PATH.write_text(json.dumps(config))     \* write the WHOLE file back
\*
\* Because the write replaces the entire document rather than merging, any
\* update another process made between our read and our write is discarded.
\*
\* Process i models a yojit invocation that has model i installed and is now
\* syncing.  Model 0 stands for an entry that was already in the file.
EXTENDS Integers, TLC

CONSTANT N
Procs == 1..N
AllModels == {0} \cup Procs

VARIABLES file, pc, snap

vars == <<file, pc, snap>>

Init ==
  /\ file = {0}
  /\ pc = [i \in Procs |-> "start"]
  /\ snap = [i \in Procs |-> {}]

Read(i) ==
  /\ pc[i] = "start"
  /\ snap' = [snap EXCEPT ![i] = file]
  /\ pc' = [pc EXCEPT ![i] = "mutate"]
  /\ UNCHANGED file

Mutate(i) ==
  /\ pc[i] = "mutate"
  /\ snap' = [snap EXCEPT ![i] = @ \cup {i}]
  /\ pc' = [pc EXCEPT ![i] = "write"]
  /\ UNCHANGED file

Write(i) ==
  /\ pc[i] = "write"
  /\ file' = snap[i]
  /\ pc' = [pc EXCEPT ![i] = "done"]
  /\ UNCHANGED snap

Next == \E i \in Procs : Read(i) \/ Mutate(i) \/ Write(i)

Spec == Init /\ [][Next]_vars

\* Once every writer has finished, the file must list every installed model.
SyncComplete ==
  (\A i \in Procs : pc[i] = "done") => (file = AllModels)

====
