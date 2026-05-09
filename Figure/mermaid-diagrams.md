# Mermaid Diagrams - Recreated from Figure Images

## 1. Traditional Memory System (intro-a.jpg)

```mermaid
flowchart LR
    Environment[🌍 Environment]
    Agent[🤖 LLM Agents]
    Memory[💾 Memory]

    Environment <-->|Interaction| Agent
    Agent -->|Write| Memory
    Memory -->|Read| Agent
```

**Traditional Memory System:**
- Environment interacts bidirectionally with LLM Agents
- LLM Agents write to Memory (simple storage)
- LLM Agents read from Memory (simple retrieval)
- Memory is passive (just stores data)

---

## 2. Agentic Memory System (intro-b.jpg)

```mermaid
flowchart LR
    Environment[🌍 Environment]
    Agent[🤖 LLM Agents]
    MemoryAgent[🤖 Memory Agent]
    Memory1[💾]
    Memory2[💾]
    Memory3[💾]

    Environment <-->|Interaction| Agent
    Agent -->|Write| MemoryAgent
    MemoryAgent -->|Read| Agent
    
    MemoryAgent <--> Memory1
    MemoryAgent <--> Memory2
    MemoryAgent <--> Memory3
```

**Agentic Memory System:**
- Environment interacts bidirectionally with LLM Agents
- LLM Agents write to Agentic Memory
- Agentic Memory reads back to LLM Agents
- **Key Difference**: Inside Agentic Memory, there's an active Memory Agent that:
  - Manages individual memories
  - Creates bidirectional relationships between memories
  - Actively organizes and evolves memories

---

## 3. Complete Agentic Memory Framework (framework.jpg)

```mermaid
flowchart TB
    subgraph NC[Note Construction]
        E1[🌍 Environment] <-->|Interaction| A1[🤖 LLM Agents]
        A1 -->|Write| C1[Conversation 1]
        A1 -->|Write| C2[Conversation 2]
        C1 --> L1[ LLM]
        C2 --> L2[🔄 LLM]
        L1 --> N1[📓 Note]
        L2 --> N2[📓 Note]
        Attr[Note Attributes] -.-> N1
        Attr -.-> N2
    end

    subgraph LG[Link Generation]
        Mem[Memory] --> B1[Box 1]
        Mem --> Bi[Box i]
        Mem --> Bj[Box j]
        Mem --> Bn[Box n]
        B1 --- Bi
        Bj --- Bn
        Bi -->|Top-k| Mj[mj]
        Mj --> Emb1[Emb1]
        Mj --> Emb2[Emb2]
        Mj --> Emb3[Emb3]
        Emb1 --> Note1[📓]
        Emb2 --> Note2[📓]
        Emb3 --> Note3[📓]
        Note1 --- Note2 --- Note3
        Note1 --> LLM1[🔄 LLM]
        Note2 --> LLM1
        Note3 --> LLM1
        LLM1 -->|Store| NB1[Box n+1]
        LLM1 -->|Store| NB2[Box n+2]
    end

    subgraph ME[Memory Evolution]
        Boxes[Box n+1, Box n+2] --> LLM2[🔄 LLM]
        LLM2 -->|Action| Evolve[⚙️ Evolve]
    end

    subgraph MR[Memory Retrieval]
        Q[Query] --> TM[Text Model]
        TM --> QE[Query Embedding]
        QE -->|Retrieve| M[Memory]
        M -->|Top-k| RM[Relative Memory]
        RM --> A2[🤖 LLM Agents]
    end

    NC -->|Retrieve| LG
    LG --> ME
    ME -->|Retrieve| MR
    MR -.->|Feedback| LG
```

**Complete Framework Breakdown:**

### **Part 1: Note Construction**
- Environment interacts with LLM Agents
- Conversations are processed by LLM
- LLM generates Notes with structured attributes:
  - Timestamp
  - Content
  - Context
  - Keywords
  - Tags
  - Embedding

### **Part 2: Link Generation**
- Memory is organized into Boxes (1 to n)
- Top-k similar memories are retrieved
- Embeddings compare memories
- LLM analyzes relationships
- New memories stored in new boxes (n+1, n+2)

### **Part 3: Memory Evolution**
- Memories evolve through LLM analysis
- Actions are taken to improve memory organization
- Memories are continuously refined

### **Part 4: Memory Retrieval**
- User Query → Text Model → Query Embedding
- Retrieve from Memory (Top-k)
- Return Relative Memory (1st, 2nd, etc.)
- Provide to LLM Agents

**Flow:**
Note Construction → Link Generation → Memory Evolution → Memory Retrieval
