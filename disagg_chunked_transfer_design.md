# Disaggregated Diffusion Multi-Transfer Design

Status: proposal for upstream review

## 1. Problem statement

The diffusion disaggregation runtime currently models each role boundary as one
payload transfer per logical request:

```text
request
  encoder -- one transfer --> denoiser -- one transfer --> decoder
```

This prevents a denoiser from exposing decode-ready latent chunks while it is
still computing later chunks. The target behavior is:

```text
request R
  denoiser: chunk 0 ---- chunk 1 ---- chunk 2 ---- done
  decoder:          decode 0 ---- decode 1 ---- decode 2
```

The design must support multiple ordered transfers for one request without
changing the behavior of existing one-transfer pipelines.

The first intended integration is chunked causal video generation such as
SANA-WM, but the transport abstraction must not contain SANA-WM-, video-, or
latent-specific assumptions.

## 2. Goals

1. Allow a role to produce zero or more ordered payloads for one logical
   request.
2. Allow denoiser compute and decoder compute to overlap with bounded memory.
3. Preserve request-level error, timeout, tracing, metrics, and client response
   semantics.
4. Support stateful consumers by pinning all chunks of a stream to one decoder
   instance and delivering them in order.
5. Give each payload an independent, auditable resource lifecycle.
6. Preserve the existing one-transfer path through a compatibility adapter,
   rather than duplicating it.
7. Work correctly for multi-rank denoiser/decoder instances.

## 3. Non-goals for the first feature PR

1. Streaming partial decoded output to the external client. The decoder may
   process chunks incrementally, but the existing frontend still receives one
   final `OutputBatch` per request.
2. Retrying a request after a process crash. The protocol provides duplicate
   detection within one server lifetime, not durable exactly-once delivery.
3. Allowing chunks of one stateful request to run on different decoder
   instances.
4. Replacing ZMQ, the transfer engine, or the pinned transfer buffer.
5. Adding arbitrary pipeline-parallel stage graphs. The initial graph remains
   encoder -> denoiser -> decoder.

## 4. Current constraints

The current implementation has several one-transfer assumptions:

- `RequestTracker` stores one mutually exclusive `RequestState`. It cannot
  represent denoiser-running and decoder-running at the same time.
- `DiffusionServer._transfer_state` is keyed by `request_id`.
- `DiffusionTransferManager._staged` and `_pending_receives` are keyed by
  `request_id`.
- Every transfer control message identifies only `request_id`.
- Denoiser output metadata is bundled into `transfer_done`; role completion and
  payload availability are therefore the same event.
- Sender selection and capacity release are inferred from the request's current
  state. That inference becomes ambiguous when roles overlap.
- Receiver preallocated-slot ownership is tied to coarse role completion rather
  than completion of one payload load.
- `find_timed_out()` measures only time since submission. A long stream making
  healthy progress can therefore time out.
- `GPUWorker.execute_forward()` and `PipelineExecutor.execute()` return one
  terminal value; there is no formal role-output iterator.

These constraints should be removed in the shared runtime. A model-specific
stage must not mutate orchestrator dictionaries or attach transport callbacks
to `Req`.

## 5. Terminology and identity

### 5.1 Logical request

The externally submitted unit, identified by `request_id`. It owns the client
reply, terminal error, overall metrics, and cancellation.

### 5.2 Transfer sequence

An ordered sequence of payloads produced for one request by a role. In the
initial implementation, only denoiser -> decoder is multi-transfer. The
encoder -> denoiser handoff remains a singleton.

There is no wire-level `stream_id`. Existing long-lived ZMQ connections are
already separated by source role and destination instance, and the pipeline
topology fixes the next role. `DiffusionServer` therefore derives source and
target from the connection on which a message arrived and the route selected
for that connection.

The connections are shared by many concurrent requests, so connection identity
does not replace `request_id` or `transfer_id`; it only removes redundant route
identity from each message.

### 5.3 Transfer

One independently staged and pushed payload in a request's transfer sequence.
It has:

```python
request_id: str
transfer_id: str
sequence_no: int
end_of_stream: bool
```

`sequence_no` starts at zero and is contiguous within the denoiser output
sequence for a request.
`transfer_id` is opaque to transport users and globally unique for the server
lifetime. A deterministic value derived from request, stream, and sequence is
acceptable, but code must compare the field rather than parse it.

The protocol deliberately uses `sequence_no`, not `chunk_idx`: payloads may be
audio blocks, refinement outputs, or other incremental units.

## 6. State model

### 6.1 Separate request progress from transfer progress

Replace the single role-wide state inference with orthogonal progress fields:

```python
class RoleStatus(Enum):
    NOT_STARTED
    WAITING
    RUNNING
    DONE

@dataclass
class TransferSequenceProgress:
    producer_instance: int | None
    consumer_instance: int | None
    next_expected_sequence: int
    final_sequence: int | None
    producer_done: bool
    decoder_completed_through: int
    queued_transfer_ids: deque[str]
    active_transfer_id: str | None
    available_credits: int

@dataclass
class RequestRecord:
    request_id: str
    encoder: RoleStatus
    denoiser: RoleStatus
    decoder: RoleStatus
    denoiser_output: TransferSequenceProgress
    terminal_status: RequestTerminalStatus | None
    submit_time: float
    last_progress_time: float
```

The exact classes may differ, but the following state must be representable:

```text
denoiser = RUNNING
decoder  = RUNNING
chunk 0  = decoder compute
chunk 1  = transfer push
chunk 2  = not produced yet
```

Existing `RequestState` may remain temporarily as a derived compatibility view,
but it must no longer drive transfer routing or resource release.

### 6.2 Per-transfer state

`DiffusionServer` stores `TransferRecord` by `transfer_id`:

```python
class TransferPhase(Enum):
    STAGED
    ALLOCATING
    PUSHING
    READY
    RECEIVED
    CONSUMING
    COMPLETED
    FAILED
    CANCELLED
```

`TransferRecord` owns sender/receiver instance IDs, pool addresses, manifest,
scalar fields, preallocated slot ID, and one-shot resource-release flags.

All cleanup operations must be idempotent. Capacity and slots are released by
the record that acquired them, never by inspecting the request's current role
state.

### 6.3 Request completion

A request succeeds only when all are true:

1. The producer has sent role-level `DONE`.
2. The request has observed exactly one `end_of_stream=True` denoiser transfer.
3. Every sequence through the final sequence has completed decoder processing.
4. The decoder has produced the final `OutputBatch`.

Receiving `DONE` early does not complete the request; it only closes the
producer side of the transfer sequence.

## 7. Wire protocol

### 7.1 Common descriptor

Add the payload identity fields to every data-plane control message. Source
role comes from the role-specific result connection, and target role comes from
the fixed route for that connection. Store both on the server-side
`TransferRecord`; do not infer them from the request's mutable progress state.

```python
@dataclass(frozen=True)
class TransferDescriptor:
    request_id: str
    transfer_id: str
    sequence_no: int
    end_of_stream: bool
```

### 7.2 Message responsibilities

Keep payload availability separate from role completion:

| Message | Direction | Responsibility |
|---|---|---|
| `transfer_staged` | producer -> server | One payload is staged and immutable |
| `transfer_alloc` | server -> consumer | Allocate receive storage for this transfer |
| `transfer_allocated` | consumer -> server | Return this transfer's receive address |
| `transfer_push` | server -> producer | Push this transfer to the supplied address |
| `transfer_pushed` | producer -> server | RDMA push finished; sender slot may be freed |
| `transfer_ready` | server -> consumer | Payload bytes are ready to load |
| `transfer_received` | consumer -> server | H2D load finished; receive slot may be recycled |
| `transfer_consumed` | consumer -> server | Decoder processing finished; return one credit |
| `role_done` | role -> server | This role will produce no more transfers |
| `request_cancel` | server -> role | Stop work and release request-owned state |

`transfer_done` should be deprecated as an overloaded event. During migration,
the single-output adapter can translate the old denoiser behavior into one
`transfer_staged(end_of_stream=True)` followed by `role_done`.

### 7.3 Decoder result

The decoder result envelope carries:

```python
request_id
transfer_id
sequence_no
end_of_stream
error
has_final_output
```

The first PR keeps the external response non-streaming. A chunk consumer may
update decoder-local state for non-final chunks and sends the final
`OutputBatch` only when `has_final_output=True` on the final chunk. Combining
video/audio/file outputs remains a pipeline responsibility, because the
transport cannot safely infer how arbitrary `OutputBatch` values compose.

## 8. Backpressure and scheduling

### 8.1 Credit window

Each request's denoiser output sequence has a bounded credit window:

```text
max_inflight_transfers_per_request = W
```

The producer must acquire a credit before staging a payload. A credit is
returned on `transfer_consumed`, not merely on `transfer_pushed`, so the bound
covers sender staging, orchestrator queues, receiver storage, and decoder work.

For useful denoiser/decoder overlap, `W >= 2`. The default can remain 1 for a
conservative first rollout, with chunked pipelines explicitly requesting 2,
or the server default can be 2 after stress tests establish memory bounds.

Allocation failure is an error, not a backpressure mechanism. Credits must stop
the producer before `stage_tensors_async()` exhausts the pool.

### 8.2 Decoder affinity and ordering

The first payload selects a decoder instance. All later payloads in that stream
are pinned to it. The decoder instance is not permanently counted as busy:

- decrement decoder capacity when dispatching one chunk;
- increment it when that chunk sends `transfer_consumed`/result;
- dispatch the next sequence for the stream only after the previous sequence is
  consumed.

This preserves causal VAE state while allowing the instance to interleave
chunks from other requests.

The orchestrator maintains a FIFO per stream and validates contiguous sequence
numbers. Out-of-order arrivals may be buffered within the credit bound, but
decoder dispatch is always ordered. Duplicate `transfer_id` or duplicate
sequence numbers fail the request with a protocol error.

### 8.3 Denoiser capacity

A denoiser slot belongs to the logical request from denoiser dispatch through
`role_done`, not through the first pushed chunk. It is released exactly once
when producer computation ends or the request is cancelled.

### 8.4 Fairness

The global decoder ready queue should schedule requests round-robin, while each
request remains FIFO. A long video request must not monopolize the
decoder pool merely because all of its credits are ready.

## 9. Buffer ownership

Every staged and receive entry in `DiffusionTransferManager` must be keyed by
`transfer_id`:

```python
_staged: dict[str, StagedTransfer]
_pending_receives: dict[str, PendingReceive]
```

The value still records `request_id` for bulk cancellation and metrics.

Ownership rules:

1. Producer staging slot: allocated before `transfer_staged`, immutable until
   push completion, freed on `transfer_pushed`.
2. Receiver slot: allocated or claimed for one transfer, retained through H2D
   load, freed/recycled on `transfer_received`.
3. Decoder GPU tensor: owned by decoder work until `transfer_consumed`.
4. Request cancellation scans transfer records by request and releases each
   outstanding resource idempotently.

Preallocated slots must be recycled per transfer on `transfer_received`, not on
coarse denoiser/decoder role completion.

## 10. Producer execution API

Do not attach a transport callback or ad-hoc fields to `Req`. Introduce a formal
role-output iterator:

```python
@dataclass
class RoleOutputChunk:
    tensor_fields: dict[str, TensorLike]
    scalar_fields: dict[str, JSONValue]
    sequence_no: int
    end_of_stream: bool

class ComposedPipelineBase:
    def iter_role_outputs(
        self, batch: Req, server_args: ServerArgs
    ) -> Iterator[RoleOutputChunk]:
        # Compatibility path
        result = self.forward(batch, server_args)
        yield RoleOutputChunk.from_req(result, sequence_no=0, end_of_stream=True)
```

Names are provisional, but the contract is not:

- existing pipelines produce one terminal chunk through the default adapter;
- chunk-capable pipelines override a documented method;
- each yielded tensor payload is immutable after yield;
- the producer declares exactly one final chunk;
- transport metadata is created by the scheduler, not by model code.

`GPUWorker.execute_forward_stream()` owns session attach, tracing, metrics, and
iteration. `_disagg_denoiser_compute()` consumes the iterator, obtains credits,
stages each payload, and sends `role_done` after exhaustion.

### 10.1 Outgoing control serialization

Multiple async D2H completions must not call the same ZMQ socket from arbitrary
threads. Add one outgoing-control queue and one socket-owning sender thread per
role instance. Compute and transfer completion threads enqueue encoded control
messages; only the sender thread touches the result PUSH socket.

For each chunk:

1. Acquire request-sequence credit.
2. Call `stage_tensors_async()`.
3. Enqueue a completion item containing the CUDA event.
4. Continue denoising when the model and buffer lifetime allow it.
5. The completion worker waits for the event and enqueues `transfer_staged` to
   the socket-owning sender.

This avoids synchronizing the compute thread on every D2H event.

### 10.2 Multi-rank correctness

All ranks in a denoiser instance must observe the same yield boundaries and
sequence numbers. Rank 0 alone stages and sends the payload, but credit waits
and cancellation decisions must be broadcast to the other ranks at each yield
boundary. Rank 0 must not block on a credit while peer ranks enter the next
collective.

Add an explicit cross-rank command at each boundary:

```text
CONTINUE(sequence_no)
WAIT
CANCEL(error)
```

Tests must cover a multi-rank producer with at least two yields. Pipelines that
cannot guarantee identical yield structure across ranks must reject chunked
handoff at initialization.

## 11. Consumer execution API

The decoder receives a fresh transport payload for each chunk but needs
request-scoped state. Add a decoder request cache keyed by `request_id` and
pinned to the selected decoder instance.

A formal consumer method should receive the descriptor and reconstructed
payload:

```python
consume_role_chunk(descriptor, req) -> DecoderChunkResult
```

The pipeline owns causal VAE cache, output accumulation/materialization, and
final output construction. The scheduler owns ordering, capacity, errors, and
transport acknowledgements.

Consumer state is released on final success, request cancellation, timeout, or
error.

## 12. SANA-WM integration

SANA-WM demonstrates two related but distinct modes.

### 12.1 Realtime ticks

The current realtime pipeline processes one externally submitted tick at a
time. Each tick already has its own `request_id`, while
`realtime_session_id` links persistent DiT and VAE state. That mode can use the
existing single-transfer adapter, provided role affinity and instance affinity
are fixed. It does not require multiple transfers within one request.

### 12.2 Offline one-request async VAE

For one request that internally denoises multiple chunks, SANA-WM implements
`iter_role_outputs()` and yields decode-ready deltas.

Important rules:

- Yield only newly decode-ready latents, never the growing full latent buffer.
- If the refiner is disabled, a stage-1 denoise chunk may be decode-ready.
- If the refiner is enabled, yield newly completed refined blocks; the transfer
  boundary is not necessarily the stage-1 chunk boundary.
- Carry absolute latent start/end indices in scalar metadata and validate them
  against the decoder frontier.
- Keep DiT/refiner state on the pinned denoiser instance.
- Keep causal VAE conv cache and output frontier on the pinned decoder instance.
- The decoder may append decoded chunks to a request-local materializer and
  produce one final `OutputBatch` at end of stream.

SANA-WM realtime chain stages also need explicit role affinities rather than the
current `MONOLITHIC` base affinity. Required cross-role modules must be declared
through the existing pipeline-specific module allow-list, not loaded as a side
effect.

## 13. Timeout, cancellation, and failure

### 13.1 Progress timeout

Track both:

- total elapsed time, for an optional hard request deadline;
- `last_progress_time`, updated by valid staged/received/consumed/role-done
  events, for the disaggregation idle timeout.

Repeated or invalid messages do not refresh progress.

### 13.2 Failure propagation

Any transfer, producer, or consumer error fails the logical request and sends
`request_cancel` to every role instance that owns request state. Cleanup must:

1. stop new producer yields;
2. discard queued decoder transfers;
3. free sender and receiver buffer entries;
4. recycle preallocated slots;
5. restore role capacities exactly once;
6. release decoder/denoiser stream state;
7. send one client error response.

Late messages for a terminal request are logged at debug/warning level and
handled idempotently; they never recreate state or release capacity twice.

## 14. Capability and compatibility

Extend role registration with:

```python
protocol_version: int
capabilities: list[str]  # includes "multi_transfer_v1"
```

A pipeline declaring chunked role output requires all selected peers to expose
the capability. The server fails startup or request admission with a clear
message instead of falling back silently.

Existing pipelines use the default iterator adapter:

```text
sequence_no = 0
end_of_stream = True
```

Their externally visible behavior and capacity defaults remain unchanged.

## 15. Metrics and tracing

Add transfer-level metrics without treating chunks as requests:

- staged/received/consumed transfer counters;
- active transfers and per-request transfer queue depth;
- producer credit wait duration;
- transfer bytes and D2H/RDMA/H2D latency;
- decoder chunk compute latency;
- cancellations and protocol errors;
- end-to-end overlap ratio or denoiser/decoder busy timelines in tests.

Trace spans include `request_id`, `transfer_id`, and `sequence_no`. Request
completion metrics fire once, after the final decoder
result.

## 16. Code change map

Expected shared-runtime changes:

- `runtime/disaggregation/transport/protocol.py`
  - descriptor fields, received/consumed/role-done/cancel messages,
    registration capabilities.
- `runtime/disaggregation/transport/manager.py`
  - key staged/receive maps by transfer ID; bulk request cleanup.
- `runtime/disaggregation/transport/buffer.py`
  - record both request and transfer identity for diagnostics.
- `runtime/disaggregation/request_state.py`
  - orthogonal role/transfer-sequence progress and progress timestamps.
- `runtime/disaggregation/orchestrator.py`
  - per-transfer records, per-request FIFO, decoder affinity, credit accounting,
    idempotent cleanup, final completion predicate.
- `runtime/disaggregation/scheduler_mixin.py`
  - role-output iteration, async staging completion, outgoing socket owner,
    credit/cancel handling, per-chunk decoder execution.
- `runtime/pipelines_core/composed_pipeline_base.py`
  - default role-output iterator contract.
- `runtime/managers/gpu_worker.py`
  - stream execution wrapper with session/tracing/multi-rank lifecycle.

Expected SANA-WM changes should be isolated to its pipeline/stages after the
shared runtime exists:

- explicit stage role affinities and module ownership;
- decode-ready delta producer;
- decoder stream state/materializer;
- parity and overlap tests.

## 17. Test plan

### 17.1 Protocol and manager unit tests

1. Encode/decode all descriptor fields.
2. Default single-transfer adapter emits sequence 0/final.
3. Stage two transfers for one request without overwriting either entry.
4. Push/free transfers independently and in reverse completion order.
5. Allocate/load/free multiple receives for one request.
6. Bulk cancellation frees all request-owned entries idempotently.

### 17.2 Request/stream state unit tests

1. Represent simultaneous denoiser and decoder activity.
2. Accept contiguous sequences and one final marker.
3. Reject duplicate transfer IDs, duplicate sequences, gaps after close, and
   multiple final markers.
4. Complete only after producer done and final decoder result.
5. Progress events refresh idle timeout; duplicates do not.

### 17.3 Orchestrator tests with fake sockets/transfer engine

1. Three chunks for one request traverse independently.
2. All chunks remain pinned to one decoder instance.
3. Decoder capacity is acquired/released per chunk; denoiser capacity once per
   request.
4. Credit window never exceeds the configured bound.
5. Per-request FIFO plus cross-request round-robin fairness.
6. Fast preallocated and slow allocation paths both recycle every slot.
7. Failure at staging, allocation, push, receive, or decode cleans every
   resource and sends one error.
8. Cancellation with queued and active chunks is idempotent.
9. Late messages after completion cannot double-release resources.

### 17.4 Worker and multi-rank tests

1. Existing pipeline uses one-output compatibility iterator.
2. Fake chunk producer yields three outputs and one role-done event.
3. Rank 0 only stages data while all ranks observe identical yield/continue
   decisions.
4. Credit wait and cancellation do not cause collective mismatch.

### 17.5 SANA-WM tests

1. Role filtering builds the intended encoder/denoiser/decoder stage lists.
2. Denoiser sends deltas, not the growing full buffer.
3. Refiner-enabled output occurs only at complete refined block boundaries.
4. Decoder validates absolute frontier and preserves conv cache.
5. Final output matches synchronous monolithic output within the existing
   parity tolerance.
6. Artificial denoise/decode delays demonstrate real overlap for window 2 and
   no overlap for window 1.
7. Two concurrent requests preserve ordering, affinity, and fairness.

### 17.6 End-to-end gate

Run a disaggregated SANA-WM request with at least three internal chunks on
separate denoiser and decoder GPUs. Verify:

- output parity with the synchronous path;
- decoder chunk 0 starts before denoiser finishes the final chunk;
- transfer and decoder queues remain within configured bounds;
- all transfer buffers, capacities, and session states return to baseline;
- cancellation midway also returns them to baseline.

## 18. Recommended upstream PR sequence

This is safer as a short dependent PR series than one model-specific patch.

### PR 1: transfer identity and lifecycle refactor (behavior preserving)

- Add `TransferDescriptor` and key all transfer resources by `transfer_id`.
- Record source/target roles on `TransferRecord` from connection context and
  fixed routing, without adding redundant wire fields.
- Add per-transfer receive acknowledgement and idempotent cleanup.
- Adapt every existing request to one final transfer.
- Add protocol/manager/orchestrator regression tests.

No pipeline emits more than one transfer in this PR.

### PR 2: multi-transfer request sequences

- Add orthogonal role/transfer-sequence progress.
- Add per-request FIFO, decoder affinity, credit window, role-done and cancel.
- Add outgoing control serialization and async staging completion.
- Add the formal producer/consumer worker APIs and multi-rank coordination.
- Use a fake chunk pipeline for complete unit/integration coverage.

### PR 3: SANA-WM async VAE integration

- Assign role affinities and module ownership.
- Emit decode-ready refined deltas.
- Add causal decoder stream state and final materialization.
- Add parity, overlap, cancellation, and end-to-end tests.

If maintainers require one PR, use the same three sections as independently
reviewable commits and keep SANA-WM changes out of the shared-runtime commits.

## 19. Acceptance criteria

The feature is complete only when all of the following are proven:

1. One logical request transfers at least three ordered payloads from denoiser
   to decoder.
2. Denoiser and decoder are simultaneously active for that request.
3. No state table is keyed only by `request_id` where multiple live transfers
   can exist.
4. Outstanding transfers never exceed the configured credit window.
5. Stateful decoder chunks remain ordered and pinned to one instance.
6. Existing one-transfer disaggregated pipelines pass unchanged.
7. Fast and slow transfer paths release all resources on success, failure,
   timeout, and cancellation.
8. Multi-rank execution cannot diverge at yield/backpressure boundaries.
9. SANA-WM output parity and measured overlap are covered by tests.
10. Request metrics and client completion occur exactly once.

## 20. Design choices to confirm with maintainers

These choices do not block the architecture but should be resolved before code:

1. Whether the initial default credit window is 1 or 2.
2. Final names for `RoleOutputChunk`, `role_done`, and `transfer_consumed`.
3. Whether `RequestState` is removed immediately or retained as a derived
   compatibility view for one release.
4. Whether protocol capability mismatch fails at server startup or at admission
   of the first chunked request.
