---- MODULE OpencodeSyncLocked ----
\* The same read-modify-write as `OpencodeSync`, but serialized by a mutex held
\* across the whole read -> modify -> write sequence.  This file exists to show
\* that the fix restores `SyncComplete`, not merely that the bug exists.
EXTENDS Integers, TLC

CONSTANT N
Procs == 1..N
AllModels == {0} \cup Procs

VARIABLES file, pc, snap, lock

vars == <<file, pc, snap, lock>>

Init ==
  /\ file = {0}
  /\ pc = [i \in Procs |-> "start"]
  /\ snap = [i \in Procs |-> {}]
  /\ lock = 0                       \* 0 = unlocked

Acquire(i) ==
  /\ pc[i] = "start"
  /\ lock = 0
  /\ lock' = i
  /\ pc' = [pc EXCEPT ![i] = "read"]
  /\ UNCHANGED <<file, snap>>

Read(i) ==
  /\ pc[i] = "read"
  /\ lock = i
  /\ snap' = [snap EXCEPT ![i] = file]
  /\ pc' = [pc EXCEPT ![i] = "mutate"]
  /\ UNCHANGED <<file, lock>>

Mutate(i) ==
  /\ pc[i] = "mutate"
  /\ lock = i
  /\ snap' = [snap EXCEPT ![i] = @ \cup {i}]
  /\ pc' = [pc EXCEPT ![i] = "write"]
  /\ UNCHANGED <<file, lock>>

Write(i) ==
  /\ pc[i] = "write"
  /\ lock = i
  /\ file' = snap[i]
  /\ lock' = 0                    \* release
  /\ pc' = [pc EXCEPT ![i] = "done"]
  /\ UNCHANGED snap

Next == \E i \in Procs : Acquire(i) \/ Read(i) \/ Mutate(i) \/ Write(i)

Spec == Init /\ [][Next]_vars

SyncComplete ==
  (\A i \in Procs : pc[i] = "done") => (file = AllModels)

====
