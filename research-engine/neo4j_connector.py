from neo4j import GraphDatabase

class KnowledgeGraphEngine:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="crimevision_password"):
        """Initializes connection to the local Dockerized Neo4j database."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        print("🔗 Connected to Neo4j Knowledge Graph!")

    def close(self):
        self.driver.close()

    def create_evidential_link(self, case_id, evidence_filename, statement_text, sinkhorn_cost):
        """
        Creates nodes and edges in Neo4j representing the AI's analysis.
        Lower sinkhorn_cost = stronger correlation.
        """
        query = """
        // 1. Ensure the Case exists
        MERGE (c:Case {id: $case_id})
        
        // 2. Create the Visual Evidence Node
        MERGE (v:VisualEvidence {filename: $evidence_filename})
        MERGE (c)-[:CONTAINS]->(v)
        
        // 3. Create the Text Statement Node
        MERGE (t:WitnessStatement {text: $statement_text})
        MERGE (c)-[:CONTAINS]->(t)
        
        // 4. Create the Mathematical Correlation Edge (The AI's Output)
        MERGE (t)-[r:CORRELATES_TO]->(v)
        SET r.sinkhorn_cost = $sinkhorn_cost,
            r.flagged_contradiction = CASE WHEN $sinkhorn_cost > 1.15 THEN true ELSE false END
            
        RETURN v, t, r
        """
        
        with self.driver.session() as session:
            result = session.run(
                query, 
                case_id=case_id, 
                evidence_filename=evidence_filename, 
                statement_text=statement_text, 
                sinkhorn_cost=sinkhorn_cost
            )
            return result.single()

# Testing the connection (if you run this script directly)
if __name__ == "__main__":
    try:
        kg = KnowledgeGraphEngine()
        print("✅ Graph Engine Ready.")
        kg.close()
    except Exception as e:
        print(f"❌ Connection Failed: {e}")
        print("Make sure your Neo4j Docker container is running!")