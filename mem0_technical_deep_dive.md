# Mem0 Technical Deep Dive: LLM Invocations, Embeddings, Vector DB, and Graph Usage

## Executive Summary

This document provides a comprehensive technical analysis of Mem0's architecture, focusing on where and how LLM invocations, embedding calls, vector database operations, and graph usage occur across different use cases: **Agentic**, **RAG**, and **Normal Prompt** scenarios.

## Architecture Overview

Mem0 implements a sophisticated memory system with the following core components:

- **LLM Layer**: Multiple provider support (OpenAI, Anthropic, Azure, AWS Bedrock, etc.)
- **Embedding Layer**: Vector representation generation for semantic search
- **Vector Store Layer**: 20+ vector database providers (Pinecone, Qdrant, Chroma, etc.)
- **Graph Store Layer**: Knowledge graph support (Neo4j, Memgraph)
- **Memory Layer**: Orchestration and storage management

---

## 1. LLM Invocation Patterns

### 1.1 Core LLM Invocation Points

#### **Primary Invocation: `Memory.add()` Method**
```python
# Location: mem0/memory/main.py:300-400
def _add_to_vector_store(self, messages, metadata, filters, infer):
    if infer:  # Most common path
        # LLM INVOCATION #1: Fact Extraction
        response = self.llm.generate_response(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        
        # LLM INVOCATION #2: Memory Update Decision
        response = self.llm.generate_response(
            messages=[{"role": "user", "content": function_calling_prompt}],
            response_format={"type": "json_object"},
        )
```

#### **Graph Memory LLM Invocations**
```python
# Location: mem0/memory/graph_memory.py:150-200
class MemoryGraph:
    def _retrieve_nodes_from_data(self, data, filters):
        # LLM INVOCATION #3: Entity Extraction
        search_results = self.llm.generate_response(
            messages=[...],
            tools=[EXTRACT_ENTITIES_TOOL]
        )
    
    def _establish_nodes_relations_from_data(self, data, filters, entity_type_map):
        # LLM INVOCATION #4: Relationship Extraction
        extracted_entities = self.llm.generate_response(
            messages=[...],
            tools=[RELATIONS_TOOL]
        )
    
    def _get_delete_entities_from_search_output(self, search_output, data, filters):
        # LLM INVOCATION #5: Deletion Decision
        memory_updates = self.llm.generate_response(
            messages=[...],
            tools=[DELETE_MEMORY_TOOL_GRAPH]
        )
```

### 1.2 LLM Provider Architecture

**Base Implementation Pattern:**
```python
# Location: mem0/llms/base.py
class LLMBase(ABC):
    @abstractmethod
    def generate_response(self, messages, tools=None, tool_choice="auto"):
        pass
```

**Provider-Specific Implementations:**
- **OpenAI**: `mem0/llms/openai.py` - Supports function calling, structured outputs
- **Anthropic**: `mem0/llms/anthropic.py` - Claude integration
- **Azure OpenAI**: `mem0/llms/azure_openai.py` - Enterprise deployment
- **AWS Bedrock**: `mem0/llms/aws_bedrock.py` - Multi-model support (Claude, Llama)

### 1.3 Edge Cases in LLM Invocations

#### **Error Handling Pattern:**
```python
try:
    response = self.llm.generate_response(...)
    response = remove_code_blocks(response)
    new_retrieved_facts = json.loads(response)["facts"]
except Exception as e:
    logging.error(f"Error in new_retrieved_facts: {e}")
    new_retrieved_facts = []
```

#### **Token Limit Handling:**
- **Chunking Strategy**: Large conversations split into manageable chunks
- **Context Window Management**: Recent messages prioritized
- **Fallback Mechanisms**: Graceful degradation when token limits exceeded

#### **Rate Limiting Edge Cases:**
- **Exponential Backoff**: Automatic retry with increasing delays
- **Provider Switching**: Fallback to alternative LLM providers
- **Batch Processing**: Multiple memories processed in single requests where possible

---

## 2. Embedding Call Patterns

### 2.1 Core Embedding Generation Points

#### **Memory Addition Flow:**
```python
# Location: mem0/memory/main.py:350-380
for new_mem in new_retrieved_facts:
    # EMBEDDING CALL #1: New Memory Encoding
    messages_embeddings = self.embedding_model.embed(new_mem, "add")
    new_message_embeddings[new_mem] = messages_embeddings
    
    # EMBEDDING CALL #2: Similarity Search
    existing_memories = self.vector_store.search(
        query=new_mem,
        vectors=messages_embeddings,
        limit=5,
        filters=filters,
    )
```

#### **Search Operations:**
```python
# Location: mem0/memory/main.py:688-726
def _search_vector_store(self, query, filters, limit, threshold=None):
    # EMBEDDING CALL #3: Query Encoding
    query_vector = self.embedding_model.embed(query, "search")
    
    # Vector similarity search
    memories = self.vector_store.search(
        query=query,
        vectors=query_vector,
        limit=limit,
        filters=filters,
    )
```

#### **Graph Memory Embeddings:**
```python
# Location: mem0/memory/graph_memory.py:400-500
def _search_graph_db(self, node_list, filters, limit=100):
    for node in node_list:
        # EMBEDDING CALL #4: Node Similarity Search
        n_embedding = self.embedding_model.embed(node)
        
        # Vector similarity in graph context
        cypher_query = f"""
        MATCH (n {self.node_label})
        WHERE n.embedding IS NOT NULL
        WITH n, vector.similarity.cosine(n.embedding, $n_embedding) AS similarity
        WHERE similarity >= $threshold
        """
```

### 2.2 Embedding Provider Architecture

#### **Base Implementation:**
```python
# Location: mem0/embeddings/base.py
class EmbeddingBase(ABC):
    @abstractmethod
    def embed(self, text, memory_action=None):
        pass
```

#### **Provider Implementations:**
- **OpenAI**: `text-embedding-3-large`, `text-embedding-3-small`
- **Azure OpenAI**: Enterprise embedding models
- **Hugging Face**: Open-source model support
- **VertexAI**: Google's embedding models

### 2.3 Embedding Edge Cases

#### **Text Preprocessing:**
```python
# Location: mem0/embeddings/openai.py:40-50
def embed(self, text, memory_action=None):
    text = text.replace("\n", " ")  # Newline normalization
    return self.client.embeddings.create(
        input=[text], 
        model=self.config.model, 
        dimensions=self.config.embedding_dims
    ).data[0].embedding
```

#### **Dimension Mismatch Handling:**
- **Model Switching**: Automatic fallback for dimension compatibility
- **Dimension Validation**: Runtime checks for embedding consistency
- **Legacy Support**: Backward compatibility for different embedding sizes

#### **Memory Action Context:**
- **"add"**: Optimized for storage operations
- **"search"**: Optimized for retrieval operations  
- **"update"**: Balanced for modification scenarios

---

## 3. Vector Database Usage Patterns

### 3.1 Core Vector Operations

#### **Insertion Pattern:**
```python
# Location: mem0/vector_stores/base.py + implementations
def insert(self, vectors, payloads=None, ids=None):
    # Batch processing for efficiency
    items = []
    for idx, vector in enumerate(vectors):
        item = {
            "id": str(ids[idx]) if ids else str(idx),
            "values": vector,
            "metadata": payloads[idx] if payloads else {}
        }
        items.append(item)
        
        if len(items) >= self.batch_size:
            self.index.upsert(vectors=items)
            items = []
```

#### **Search Pattern:**
```python
# Location: mem0/vector_stores/pinecone.py:190-230
def search(self, query, vectors, limit=5, filters=None):
    filter_dict = self._create_filter(filters) if filters else None
    
    query_params = {
        "vector": vectors,
        "top_k": limit,
        "include_metadata": True,
        "include_values": False,
    }
    
    if filter_dict:
        query_params["filter"] = filter_dict
        
    # Hybrid search support
    if self.hybrid_search and self.sparse_encoder:
        sparse_vector = self.sparse_encoder.encode_queries(query_text)
        query_params["sparse_vector"] = sparse_vector
```

### 3.2 Provider-Specific Implementations

#### **Pinecone Integration:**
- **Serverless/Pod Configuration**: Flexible deployment options
- **Hybrid Search**: Dense + sparse vector combination
- **Metadata Filtering**: Complex filter expressions
- **Batch Operations**: Optimized for high throughput

#### **Qdrant Integration:**
- **Local/Cloud Deployment**: Flexible hosting options
- **Payload Indexing**: Fast metadata filtering
- **Quantization**: Memory-efficient vector storage

#### **Chroma Integration:**
- **Lightweight Deployment**: Suitable for development/testing
- **Document Store**: Built-in text storage capabilities

### 3.3 Vector Database Edge Cases

#### **Index Management:**
```python
# Location: mem0/vector_stores/pinecone.py:90-120
def create_col(self, vector_size, metric="cosine"):
    existing_indexes = self.list_cols().names()
    
    if self.collection_name in existing_indexes:
        logging.debug(f"Index {self.collection_name} already exists")
        self.index = self.client.Index(self.collection_name)
        return
    
    # Create new index with appropriate configuration
    spec = ServerlessSpec(cloud="aws", region="us-west-2")
    self.client.create_index(
        name=self.collection_name,
        dimension=vector_size,
        metric=metric,
        spec=spec,
    )
```

#### **Filter Complexity:**
```python
def _create_filter(self, filters):
    pinecone_filter = {}
    for key, value in filters.items():
        if isinstance(value, dict) and "gte" in value and "lte" in value:
            pinecone_filter[key] = {"$gte": value["gte"], "$lte": value["lte"]}
        else:
            pinecone_filter[key] = {"$eq": value}
    return pinecone_filter
```

#### **Connection Resilience:**
- **Retry Logic**: Automatic reconnection on failures
- **Connection Pooling**: Efficient resource utilization
- **Health Checks**: Proactive connection monitoring

---

## 4. Graph Database Usage Patterns

### 4.1 Core Graph Operations

#### **Entity and Relationship Addition:**
```python
# Location: mem0/memory/graph_memory.py:80-150
def add(self, data, filters):
    # Step 1: Extract entities from text
    entity_type_map = self._retrieve_nodes_from_data(data, filters)
    
    # Step 2: Establish relationships
    to_be_added = self._establish_nodes_relations_from_data(data, filters, entity_type_map)
    
    # Step 3: Search existing graph
    search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
    
    # Step 4: Determine deletions
    to_be_deleted = self._get_delete_entities_from_search_output(search_output, data, filters)
    
    # Step 5: Execute operations
    deleted_entities = self._delete_entities(to_be_deleted, filters)
    added_entities = self._add_entities(to_be_added, filters, entity_type_map)
```

#### **Neo4j Cypher Queries:**
```cypher
-- Node creation with vector embeddings
MERGE (source:__Entity__ {name: $source_name, user_id: $user_id})
ON CREATE SET
    source.created = timestamp(),
    source.mentions = 1
ON MATCH SET
    source.mentions = coalesce(source.mentions, 0) + 1
CALL db.create.setNodeVectorProperty(source, 'embedding', $source_embedding)

-- Relationship creation
MERGE (source)-[r:RELATIONSHIP_TYPE]->(destination)
ON CREATE SET 
    r.created = timestamp(),
    r.mentions = 1
ON MATCH SET
    r.mentions = coalesce(r.mentions, 0) + 1
```

### 4.2 Graph Search Patterns

#### **Vector Similarity in Graph Context:**
```python
# Location: mem0/memory/graph_memory.py:300-350
def _search_graph_db(self, node_list, filters, limit=100):
    for node in node_list:
        n_embedding = self.embedding_model.embed(node)
        
        cypher_query = f"""
        MATCH (n {self.node_label})
        WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
        WITH n, vector.similarity.cosine(n.embedding, $n_embedding) AS similarity
        WHERE similarity >= $threshold
        CALL {{
            MATCH (n)-[r]->(m) RETURN n.name AS source, type(r) AS relationship, m.name AS destination
            UNION
            MATCH (m)-[r]->(n) RETURN m.name AS source, type(r) AS relationship, n.name AS destination
        }}
        RETURN source, relationship, destination, similarity
        ORDER BY similarity DESC
        LIMIT $limit
        """
```

#### **BM25 Reranking:**
```python
# Location: mem0/memory/graph_memory.py:120-140
def search(self, query, filters, limit=100):
    entity_type_map = self._retrieve_nodes_from_data(query, filters)
    search_output = self._search_graph_db(node_list=list(entity_type_map.keys()), filters=filters)
    
    # Convert to sequences for BM25
    search_outputs_sequence = [
        [item["source"], item["relationship"], item["destination"]] 
        for item in search_output
    ]
    
    # Rerank using BM25
    bm25 = BM25Okapi(search_outputs_sequence)
    tokenized_query = query.split(" ")
    reranked_results = bm25.get_top_n(tokenized_query, search_outputs_sequence, n=5)
```

### 4.3 Graph Edge Cases

#### **Node Deduplication:**
```python
def _add_entities(self, to_be_added, filters, entity_type_map):
    # Search for existing nodes with similar embeddings
    source_node_search_result = self._search_source_node(source_embedding, filters, threshold=0.9)
    destination_node_search_result = self._search_destination_node(dest_embedding, filters, threshold=0.9)
    
    # Handle all combinations: both exist, one exists, neither exists
    if not destination_node_search_result and source_node_search_result:
        # Create destination, link to existing source
    elif destination_node_search_result and not source_node_search_result:
        # Create source, link to existing destination  
    elif source_node_search_result and destination_node_search_result:
        # Link existing nodes
    else:
        # Create both nodes and relationship
```

#### **Multi-tenant Graph Isolation:**
```python
# User isolation in all queries
agent_filter = ""
params = {"user_id": filters["user_id"]}
if filters.get("agent_id"):
    agent_filter = "AND n.agent_id = $agent_id AND m.agent_id = $agent_id"
    params["agent_id"] = filters["agent_id"]
```

---

## 5. Use Case Analysis

### 5.1 Normal Prompt Use Case

#### **Flow:**
1. **Input**: Simple text/conversation
2. **LLM Invocation**: Fact extraction from input
3. **Embedding**: Generate vectors for extracted facts
4. **Vector Search**: Find similar existing memories  
5. **LLM Invocation**: Decide on memory operations (ADD/UPDATE/DELETE)
6. **Storage**: Execute vector and optional graph operations

#### **Example Implementation:**
```python
# Basic usage
m = Memory()
result = m.add("I love playing tennis on weekends", user_id="alice")
# Output: [{"id": "uuid", "memory": "Loves playing tennis", "event": "ADD"}]

# Search
results = m.search("What does Alice enjoy?", user_id="alice")
# Output: [{"memory": "Loves playing tennis", "score": 0.85}]
```

#### **Edge Cases:**
- **Empty Facts**: No extractable information → No memory operations
- **Duplicate Detection**: Similar facts → UPDATE instead of ADD
- **Context Length**: Long conversations → Chunking and summarization

### 5.2 RAG (Retrieval-Augmented Generation) Use Case

#### **Implementation Pattern:**
```python
# Location: evaluation/src/rag.py
class RAGManager:
    def search(self, query, chunks, embeddings, k=1):
        # Embedding-based retrieval
        query_embedding = self.calculate_embedding(query)
        similarities = [self.calculate_similarity(query_embedding, embedding) 
                       for embedding in embeddings]
        
        # Top-k selection
        top_indices = np.argsort(similarities)[-k:][::-1]
        combined_chunks = "\n<->\n".join([chunks[i] for i in top_indices])
        
        return combined_chunks, search_time
    
    def generate_response(self, question, context):
        # LLM invocation with retrieved context
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant..."},
                {"role": "user", "content": f"Question: {question}\nContext: {context}"}
            ],
            temperature=0,
        )
```

#### **Flow:**
1. **Chunking**: Split documents using tiktoken
2. **Embedding**: Generate embeddings for all chunks
3. **Query Processing**: Embed user query
4. **Retrieval**: Cosine similarity search for top-k chunks
5. **LLM Generation**: Generate response using retrieved context

#### **Edge Cases:**
- **Chunk Size Optimization**: Token-aware chunking with tiktoken
- **Relevance Threshold**: Filter low-similarity chunks
- **Context Window**: Combine multiple chunks within token limits
- **No Relevant Context**: Graceful handling when no good matches found

### 5.3 Agentic Use Case

#### **AutoGen Integration:**
```python
# Location: cookbooks/mem0-autogen.ipynb
class Mem0ProxyCoderAgent(UserProxyAgent):
    def initiate_chat(self, assistant, message):
        # Retrieve agent-specific memories
        agent_memories = self.memory.search(message, agent_id=self.agent_id, limit=3)
        agent_memories_txt = "\n".join(mem["memory"] for mem in agent_memories)
        
        # Inject memories into prompt
        prompt = f"{message}\nCoding Preferences: \n{agent_memories_txt}"
        response = super().initiate_chat(assistant, message=prompt)
        
        # Store new interaction
        MEMORY_DATA = [
            {"role": "user", "content": message}, 
            {"role": "assistant", "content": response}
        ]
        self.memory.add(MEMORY_DATA, agent_id=self.agent_id)
        return response
```

#### **Procedural Memory:**
```python
# Location: mem0/memory/main.py:832-870
def _create_procedural_memory(self, messages, metadata=None, prompt=None):
    if prompt:
        system_prompt = prompt
    else:
        system_prompt = PROCEDURAL_MEMORY_SYSTEM_PROMPT
    
    user_prompt = parse_messages(messages)
    
    # LLM generates procedural knowledge
    response = self.llm.generate_response(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Store as procedural memory
    memory_id = self._create_memory(
        data=response,
        existing_embeddings={response: self.embedding_model.embed(response, "add")},
        metadata=metadata
    )
```

#### **Flow:**
1. **Agent Initialization**: Setup with agent-specific memory scope
2. **Context Retrieval**: Search relevant memories for current task
3. **Prompt Augmentation**: Inject retrieved memories into agent prompt
4. **LLM Interaction**: Generate response with memory context
5. **Memory Update**: Store new interaction for future reference
6. **Learning**: Continuous improvement through memory accumulation

#### **Edge Cases:**
- **Memory Scope**: Agent-specific vs. shared memories
- **Preference Conflicts**: Handling contradictory learned preferences
- **Memory Overflow**: Pruning strategies for long-running agents
- **Context Relevance**: Ensuring retrieved memories are task-relevant

---

## 6. Advanced Edge Cases and Error Handling

### 6.1 Concurrent Access Patterns

#### **Race Conditions:**
```python
# Memory consistency during concurrent operations
with concurrent.futures.ThreadPoolExecutor() as executor:
    future1 = executor.submit(self._add_to_vector_store, messages, metadata, filters, infer)
    future2 = executor.submit(self._add_to_graph, messages, filters)
    
    concurrent.futures.wait([future1, future2])
```

#### **Async Operations:**
```python
# Location: mem0/memory/main.py:1176-1290
async def process_fact_for_search(new_mem_content):
    messages_embeddings = await asyncio.get_event_loop().run_in_executor(
        None, self.embedding_model.embed, new_mem_content, "add"
    )
    existing_memories = await asyncio.get_event_loop().run_in_executor(
        None, self.vector_store.search, new_mem_content, messages_embeddings, 5, effective_filters
    )
```

### 6.2 Memory Consistency Edge Cases

#### **UUID Hallucination Prevention:**
```python
# Mapping UUIDs with integers for LLM processing
temp_uuid_mapping = {}
for idx, item in enumerate(retrieved_old_memory):
    temp_uuid_mapping[str(idx)] = item["id"]
    retrieved_old_memory[idx]["id"] = str(idx)
```

#### **JSON Parsing Robustness:**
```python
try:
    response = remove_code_blocks(response)
    new_retrieved_facts = json.loads(response)["facts"]
except Exception as e:
    logging.error(f"Error in new_retrieved_facts: {e}")
    new_retrieved_facts = []
```

### 6.3 Performance Edge Cases

#### **Embedding Caching:**
```python
new_message_embeddings = {}
for new_mem in new_retrieved_facts:
    messages_embeddings = self.embedding_model.embed(new_mem, "add")
    new_message_embeddings[new_mem] = messages_embeddings  # Cache for reuse
```

#### **Batch Processing:**
```python
# Vector store batch operations
if len(items) >= self.batch_size:
    self.index.upsert(vectors=items)
    items = []
```

### 6.4 Multi-tenancy Edge Cases

#### **Filter Inheritance:**
```python
def _build_filters_and_metadata(*, user_id=None, agent_id=None, run_id=None, actor_id=None):
    if not any([user_id, agent_id, run_id]):
        raise ValueError("At least one of 'user_id', 'agent_id', or 'run_id' must be provided.")
    
    # Ensure proper isolation
    effective_query_filters["user_id"] = user_id
    if agent_id:
        effective_query_filters["agent_id"] = agent_id
```

### 6.5 Graph Consistency Edge Cases

#### **Node Similarity Thresholds:**
```python
# Prevent duplicate nodes with high similarity
source_node_search_result = self._search_source_node(source_embedding, filters, threshold=0.9)
if source_node_search_result:
    # Use existing node instead of creating new one
```

#### **Relationship Deduplication:**
```python
# Prevent duplicate relationships
MERGE (source)-[r:RELATIONSHIP_TYPE]->(destination)
ON CREATE SET r.created = timestamp(), r.mentions = 1
ON MATCH SET r.mentions = coalesce(r.mentions, 0) + 1
```

---

## 7. Performance Optimization Patterns

### 7.1 Embedding Optimization

#### **Dimension Tuning:**
- **text-embedding-3-large**: 3072 dimensions (high accuracy)
- **text-embedding-3-small**: 1536 dimensions (balanced performance)
- **Custom dimensions**: Provider-specific optimizations

#### **Batch Embedding:**
```python
# Process multiple texts in single API call
embeddings = self.client.embeddings.create(
    input=text_batch,  # List of texts
    model=self.config.model
).data
```

### 7.2 Vector Search Optimization

#### **Index Configuration:**
```python
# Pinecone serverless for auto-scaling
spec = ServerlessSpec(cloud="aws", region="us-west-2")

# Qdrant quantization for memory efficiency
quantization_config = {
    "scalar": {
        "type": "int8",
        "quantile": 0.99,
        "always_ram": True
    }
}
```

#### **Hybrid Search:**
```python
# Dense + sparse vector combination
if self.hybrid_search and self.sparse_encoder:
    sparse_vector = self.sparse_encoder.encode_queries(query_text)
    query_params["sparse_vector"] = sparse_vector
```

### 7.3 Graph Optimization

#### **Index Strategy:**
```cypher
-- Composite indexes for multi-property searches
CREATE INDEX entity_composite IF NOT EXISTS 
FOR (n:__Entity__) ON (n.name, n.user_id)

-- Vector similarity indexes
CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
FOR (n:__Entity__) ON (n.embedding)
```

#### **Query Optimization:**
```cypher
-- Efficient traversal with APOC procedures
CALL apoc.path.subgraphNodes(startNode, {
    relationshipFilter: "RELATED_TO>",
    labelFilter: "__Entity__",
    maxLevel: 3
}) YIELD node
```

---

## 8. Security and Privacy Considerations

### 8.1 Data Isolation

#### **Multi-tenant Filtering:**
```python
# Ensure all queries include tenant isolation
WHERE n.user_id = $user_id AND m.user_id = $user_id
```

#### **Agent Scoping:**
```python
# Optional agent-level isolation
if filters.get("agent_id"):
    agent_filter = "AND n.agent_id = $agent_id"
```

### 8.2 Embedding Privacy

#### **Local Embedding Options:**
```python
# Hugging Face local models
embedding_config = {
    "provider": "huggingface",
    "config": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu"  # or "cuda"
    }
}
```

---

## 9. Monitoring and Observability

### 9.1 Telemetry Integration

```python
# Location: mem0/memory/telemetry.py
def capture_event(event_name, memory_instance, properties):
    # Track usage patterns
    keys, encoded_ids = process_telemetry_filters(filters)
    capture_event("mem0.add", self, {
        "version": self.api_version,
        "keys": keys,
        "encoded_ids": encoded_ids,
        "sync_type": "sync"
    })
```

### 9.2 Performance Metrics

#### **Timing Measurements:**
```python
# Search timing
t1 = time.time()
query_embedding = self.calculate_embedding(query)
# ... search operations ...
t2 = time.time()
return combined_chunks, t2 - t1
```

#### **Cost Tracking:**
- **Embedding API Calls**: Count and cost per embedding
- **LLM Token Usage**: Input/output token tracking
- **Vector Operations**: Search and storage costs

---

## 10. Migration and Versioning

### 10.1 API Version Management

```python
if self.api_version == "v1.0":
    warnings.warn(
        "The current add API output format is deprecated. "
        "To use the latest format, set `api_version='v1.1'`.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return vector_store_result

# New format
return {"results": vector_store_result, "relations": graph_result}
```

### 10.2 Schema Evolution

#### **Vector Store Migration:**
```python
# Separate migration vector store
self.config.vector_store.config.collection_name = "mem0migrations"
self._telemetry_vector_store = VectorStoreFactory.create(
    self.config.vector_store.provider, 
    self.config.vector_store.config
)
```

---

## Conclusion

Mem0's architecture demonstrates sophisticated orchestration of LLM invocations, embedding operations, vector database management, and graph operations across diverse use cases. The system handles complex edge cases through robust error handling, multi-tenancy support, and performance optimizations while maintaining flexibility for normal prompts, RAG applications, and agentic systems.

Key technical highlights:
- **5 distinct LLM invocation patterns** for different memory operations
- **4 embedding call types** optimized for specific use cases  
- **20+ vector database providers** with unified interface
- **Graph-vector hybrid approach** for rich knowledge representation
- **Comprehensive edge case handling** for production reliability
- **Multi-tenant architecture** with proper data isolation
- **Performance optimizations** across all system layers

This architecture enables Mem0 to serve as a robust foundation for building memory-augmented AI applications across various domains and scales.