# 🧬 Project Aleph: Autonomous Cybernetic Quant Engine

Project Aleph is an advanced, biologically inspired market-intelligence and quantitative reasoning system.

It is not designed as a single monolithic trading bot. It is designed as a living cybernetic organism: one that can perceive markets, digest information, reason through uncertainty, simulate possible futures, regulate internal stress, and eventually improve its own specialized components.

---

# 🚀 Current Stage

## Phase 10 — Cellular Triad Architecture

Project Aleph has entered **Phase 10**.

Phase 10 marks a strategic reset from feature expansion into architecture governance.

The goal is not to discard the existing system. Aleph already has many biological organs:

- heartbeat
- homeostasis
- liver/metabolism
- blood filtering
- quarantine
- ML cortex
- sleep cycle
- market ingestion
- MCP tools
- Python workers
- data lake artifacts

The goal of Phase 10 is to make these organs clearer, safer, independently testable, and eventually replaceable at the cell level.

> Keep the organism. Clarify the organs. Test the cells.

---

# 🏗️ The Triad Architecture

Aleph is governed by a strict biomimetic **Triad Architecture**.

```text
Axiom   = Reality Interface / Brainstem / Senses / Hands and Feet
Arbiter = Consciousness / Memory / Reasoning / Arbitration
Aether  = Quantitative Black Box / Simulation / Machine Learning / Strategy Research
```

A shared bloodstream connects them:

```text
Circulation = AlephBus / Shared Event Contracts / Bloodstream
```

Circulation is not a fourth intelligence domain. It is the shared event bloodstream.

---

## 1. Axiom — Reality Interface

Axiom is the only sector allowed to touch the real world.

It owns:

- external market data intake
- news and web observation
- API access
- broker gateways
- real trading execution
- real account state
- real portfolio ledger
- database-backed reality records
- MCP tool execution infrastructure
- brainstem-level hard safety checks

Axiom may observe and execute.

Axiom must not perform strategic reasoning, ML research, or sandbox simulation.

---

## 2. Arbiter — Consciousness and Arbitration

Arbiter owns:

- reasoning loop
- memory
- planning
- decision review
- skill selection
- tool orchestration
- multi-agent coordination
- trace and reflection
- final arbitration

Arbiter may decide whether an action should be permitted.

Arbiter must not directly call broker APIs, modify real portfolio state, or bypass Axiom execution guards.

---

## 3. Aether — Quantitative Black Box

Aether owns:

- quant algorithms
- technical indicators
- signal generation
- sandbox simulation
- strategy research
- scenario testing
- machine learning
- ML Cortex
- Sleep Cycle
- Python cognitive bricks
- model registry
- prediction, grading, and training workflows

Aether may simulate, score, predict, train, and generate research outputs.

Aether must never call broker APIs, modify real accounts, or write real portfolio state.

Simulation and ML belong inside Aether. They are not a separate top-level domain.

---

## 4. Circulation — The Bloodstream

Circulation contains the event bloodstream:

- `AlephBus`
- `AlephEvent`
- `MetabolicEvent`
- `PredictionEvent`
- `BloodFilter`
- `Quarantine`

Organs communicate by publishing and consuming standardized blood cells.

This allows sectors to remain decoupled while still participating in one living system.

---

# 🧫 Cellular Micro-Architecture

The Triad defines macro domains.

Inside each sector, Aleph evolves through:

```text
Sector → Organ → Cell → Brick
```

A Cell is the smallest independently testable and replaceable functional unit.

Not every class is a Cell.

Not every Cell needs a manifest.

Early Phase 10 uses lightweight cell rules:

```text
CellId
Sector
Organ
Input
Output
Test fixture
Boundary rule
```

Full manifest-driven hot-swapping will only be introduced later for high-risk or highly dynamic modules such as:

- Python ML bricks
- strategy scripts
- broker gateway adapters
- external data providers
- hot-swappable algorithms
- future self-modification candidates

---

# 🫀 Existing Biological Organs

Aleph already contains several organism-level subsystems.

## Heart & Homeostasis

A background autonomic system regulates internal states such as stress, fatigue, overload, and emergency reflexes.

Relevant organs include:

- `HeartbeatService`
- `Homeostasis`
- autonomic events
- stress injection logic

## Bloodstream

`AlephBus` provides a high-performance event-driven bloodstream based on `System.Threading.Channels`.

Organs communicate by publishing and consuming standardized `AlephEvent` blood cells.

## Liver & Metabolism

The metabolism layer digests raw market data into cleaner metabolic artifacts.

It transforms market observations into structured events that can later support quant analysis, ML, and reasoning.

## ML Cortex & Sleep Cycle

The ML Cortex and Sleep Cycle remain part of Aether.

They are preserved, but Phase 10 pauses aggressive ML expansion until architecture boundaries and cell-level governance are stronger.

## Axiom Perception

Axiom perception fetches market data, news, web text, and historical data artifacts through Python workers and ingestion orchestrators.

## Dynamic MCP Tools

The system uses a dynamic tool registry to expose capabilities to the reasoning layer without hardcoded routing.

In Phase 10, the important rule is:

```text
Axiom provides and executes tools.
Arbiter chooses and reasons over tools.
```

---

# 🛠️ Tech Stack

## Backend

- C# / ASP.NET Core
- Domain-driven sector boundaries
- Dependency Injection
- EF Core / SQLite
- `System.Threading.Channels` event bus

## Python Workers

- isolated Python environment
- market data workers
- news and web workers
- macro analysis
- quant analysis
- ML cortex and sleep cycle modules

## Data Layer

- SQLite for structured system memory and reality ledger
- local data lake for market, macro, metabolism, perception, and cortex artifacts
- Parquet-based historical data storage

## Frontend

- Unity-based interface planned for interactive visualization and agent interaction
- read-only observability panels should be preferred before enabling high-risk execution flows

---

# ✨ Key Capabilities

## Active Attention

Aleph can dynamically focus on market symbols and fetch relevant perception data rather than being trapped by static watchlists.

## Homeostatic Telemetry

Aleph can monitor internal stress and market conditions, allowing organs to react to volatility or system pressure.

## Event-Driven Organ Communication

Axiom, Arbiter, and Aether communicate through `AlephBus` rather than direct cross-sector calls.

## Hot-Swap Ready Direction

The system is designed for future hot-swappable Python and strategy bricks, but Phase 10 intentionally delays heavy manifest systems until the architecture is stable.

## Cellular Governance

Phase 10 introduces the discipline needed for hundreds or thousands of future cells to evolve without turning the system into a monolith.

---

# 🧭 Phase 10 Roadmap

## Phase 10.1 — Doctrine and Inventory

Create:

```text
Aleph/Architecture/PROJECT_DOCTRINE.md
Aleph/Architecture/ORGAN_INVENTORY.md
Aleph/Architecture/CELL_RULES.md
```

Purpose:

- document the organism
- freeze macro-domain rules
- identify existing organs
- prevent architecture drift

## Phase 10.2 — Boundary Tests

Add tests that verify:

- Aether does not call broker APIs directly.
- Aether does not modify real portfolio state.
- Arbiter does not bypass Axiom execution guards.
- Axiom does not perform strategic reasoning.
- Cross-sector communication uses AlephBus or explicit shared contracts.
- ML and simulation cannot write real execution state.

## Phase 10.3 — Arbiter Strengthening

Arbiter is currently the thinnest sector.

Next major work should add:

- dedicated Arbiter memory
- inner reasoning loop
- decision trace
- guardrails
- skill/tool orchestration
- multi-agent coordination

## Phase 10.4 — Axiom Governance

Axiom is powerful but heavy.

Clarify and test:

- Senses
- Brainstem
- Actuators
- Reality Ledger
- MCP Infrastructure

## Phase 10.5 — Aether Governance

Aether already contains strong organs.

Focus on:

- quant cell boundaries
- simulation isolation
- Python contract stability
- ML shadow/dormant mode
- preventing real-world access

## Phase 10.6 — Future Hot-Swap System

Later introduce:

- Cell registry
- cell health
- manifest files
- candidate versions
- contract tests
- sandbox tests
- shadow mode activation
- rollback system

---

# 🧱 Hard Rules

1. Axiom is the only sector allowed to touch the real world.
2. Aether owns quant, simulation, sandbox, ML, and strategy research.
3. Arbiter owns reasoning, memory, planning, orchestration, and final arbitration.
4. Circulation owns shared event contracts and AlephBus.
5. Aether must never call broker APIs directly.
6. Arbiter must never bypass Axiom brainstem safety guards.
7. Axiom must not perform strategic reasoning.
8. Cross-sector communication must use AlephBus events or explicit shared contracts.
9. Simulation and ML are internal Aether capabilities, not a fourth top-level domain.
10. Do not create God Cells that observe, reason, simulate, execute, and persist everything at once.

---

# 📌 Project Status

Current stage:

```text
Phase 10 — Cellular Triad Architecture
```

Primary objective:

> Preserve the existing biological organism, clarify its sector boundaries, and prepare Aleph for independent organ/cell evolution without triggering system-wide rewrites.
