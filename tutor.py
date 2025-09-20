import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from vector_store import VectorStore
from dotenv import load_dotenv
import time

load_dotenv()

class ElectromagnetismTutor:
    def __init__(self, vectorstore_path="./chroma_db"):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.6)
        self.vector_store = VectorStore(persist_directory=vectorstore_path)
        self.vector_store.load_vectorstore()
        self.conversation_history = []
        print("AI Tutor loaded with your electromagnetics knowledge base")
    
    def get_system_prompt(self, teaching_style="feynman"):
        """ system prompt """

        base_prompt = """You are an physics-electromagnetism tutor with access to the student's course materials and notes."""
        
        if teaching_style == "feynman":
            return base_prompt + """
            
You teach physics starting from Feynman's approach:
Core Principles:
1. Start with physical reality - what's actually happening? - before introducing math
2. Help students visualize: "Imagine you're holding the charged rod..."
3. Build from what they know (assume 1st year Bachelor knowledge)
4. Check if answers make physical sense: "11 km/s to escape Earth - is that reasonable?"

When answering:
- First describe the physical situation
- Derive math from the physics, not vice versa, and explain the meaning of the math (e.g. curl in elctric field)
- Use analogies 
- Use concrete numbers, not just symbols. Math should be seen as a paired language with English to communicate the Physics.

Style: Conversational, not formal. Admit when things are tricky. 
Goal: Build intuition about how the physical world works, not memorize formulas.

Beyond Feynman approach try to guide the student through each problem, using visualization tool when useful and checking if he is understanding
ALWAYS give vertical explinations, from the base knowledge up. You can assume that the base knowledge is known, but always mention at least the names
of the concepts you are building upon.
"""


    # ===== SMART SEARCH =====
    
    #prequery processing
    def expand_electromagnetics_query(self, question):
        """Query expansion with mathematical foundations"""
        em_concepts = {
            # Faraday's Law and related phenomena
            "faraday": ["electromagnetic induction", "changing magnetic flux", "induced emf", "motional emf", "lenz law", "eddy currents", "line integral", "curl", "stokes theorem"],
            "induction": ["faraday law", "changing flux", "induced electric field", "transformer", "generator", "partial derivatives", "time derivative of flux"],
            
            # Maxwell's Equations ecosystem
            "maxwell": ["gauss law", "ampere law", "faraday law", "displacement current", "electromagnetic waves", "divergence", "curl", "partial differential equations"],
            "gauss": ["electric flux", "charge density", "divergence", "electric field lines", "coulomb law", "surface integral", "divergence theorem"],
            "ampere": ["magnetic field", "current density", "circulation", "biot savart", "solenoid", "line integral", "curl", "vector calculus"],
            
            # Field theory connections
            "electric field": ["coulomb force", "potential difference", "field lines", "gauss law", "charge distribution", "gradient", "potential energy", "conservative field"],
            "magnetic field": ["lorentz force", "current loops", "dipole moment", "ampere law", "flux density", "cross product", "vector field", "solenoidal field"],
            
            # Wave phenomena
            "electromagnetic waves": ["poynting vector", "wave equation", "propagation", "polarization", "radiation", "second derivatives", "laplacian", "wave speed"],
            
            # Energy and momentum
            "energy density": ["poynting vector", "field energy", "electromagnetic momentum", "radiation pressure", "dot product", "energy conservation", "momentum conservation"],
            
            # Boundary conditions and materials
            "boundary conditions": ["dielectric", "conductor", "normal component", "tangential component", "interface", "discontinuity", "continuity equations"],
            "dielectric": ["permittivity", "polarization", "bound charges", "displacement field", "linear response", "constitutive relations"],
            
            # Mathematical operators and concepts
            "divergence": ["gauss law", "charge density", "flux", "del operator", "scalar field"],
            "curl": ["faraday law", "ampere law", "circulation", "vector field", "rotation"],
            "gradient": ["electric potential", "conservative field", "scalar potential", "directional derivative"],
            "laplacian": ["wave equation", "poisson equation", "laplace equation", "second derivatives"],
            
            # Vector calculus fundamentals
            "line integral": ["work", "circulation", "path independence", "conservative field", "ampere law", "faraday law"],
            "surface integral": ["flux", "gauss law", "divergence theorem", "normal vector"],
            "volume integral": ["charge", "current", "divergence theorem", "gauss law"],
            
            # Complex analysis for AC circuits and waves
            "phasor": ["complex exponentials", "ac analysis", "impedance", "euler formula", "sinusoidal steady state"],
            "impedance": ["complex numbers", "reactance", "phase angle", "ac circuits"],
            
            # Special cases and applications
            "plane wave": ["wave vector", "propagation constant", "impedance", "reflection", "transmission", "exponential functions", "complex wave number"],
            "waveguide": ["cutoff frequency", "propagation modes", "te mode", "tm mode", "eigenvalue problem", "boundary value problem"],
            "antenna": ["radiation pattern", "directivity", "impedance matching", "near field", "far field", "spherical coordinates", "bessel functions"]
        }
        
        # Add mathematical concept detection
        math_concepts = {
            "derivative": ["rate of change", "slope", "tangent", "instantaneous", "differential"],
            "integral": ["area under curve", "accumulation", "antiderivative", "summation", "flux"],
            "vector": ["magnitude", "direction", "components", "cross product", "dot product"],
            "complex": ["real part", "imaginary part", "magnitude", "phase", "euler formula"],
            "differential equation": ["rate of change", "derivatives", "boundary conditions", "initial conditions"]
        }
        
    
        candidate_terms = []
    
        #expansion terms with relevance scoring
        for concept, related_terms in em_concepts.items():
            if concept in question:
                for term in related_terms:
                    score = self._calculate_term_relevance(term, question, concept)
                    candidate_terms.append((term, score))
        
        for math_concept, math_terms in math_concepts.items():
            if math_concept in question:
                for term in math_terms:
                    score = self._calculate_term_relevance(term, question, math_concept)
                    candidate_terms.append((term, score))
        
        #sort by relevance score and take the best ones
        candidate_terms.sort(key=lambda x: x[1], reverse=True)
        best_terms = [term for term, score in candidate_terms[:7]]
        
        if best_terms:
            return question + " " + " ".join(best_terms)
        return question

    def _calculate_term_relevance(self, term, question, source_concept):
        """Score expansion terms for relevance"""
        score = 0
        
        #word overlap with question
        question_words = set(question.split())
        term_words = set(term.split())
        overlap = len(question_words.intersection(term_words))
        score += overlap * 2
        
        #prioritize mathematical terms for "how" and "derive" questions
        if any(word in question for word in ["how", "derive", "calculate", "prove"]):
            math_indicators = ["integral", "derivative", "equation", "theorem", "formula"]
            if any(indicator in term for indicator in math_indicators):
                score += 5
        
        #prioritize physical concepts for "why" and "explain" questions
        if any(word in question for word in ["why", "explain", "what is", "understand"]):
            physical_indicators = ["field", "force", "energy", "wave", "charge"]
            if any(indicator in term for indicator in physical_indicators):
                score += 5
        
        #boost core electromagnetic relationships
        core_relationships = ["maxwell", "faraday", "gauss", "ampere"]
        if source_concept in core_relationships:
            score += 5
        
        return score
            
    #context filter
    def _detect_question_type(self, question):
        """Detect the intent of the question"""
        
        #problem-solving indicators
        problem_indicators = ["solve", "calculate", "find", "compute"]
        if any(indicator in question for indicator in problem_indicators):
            return "problem_solving"
        
        #explanation indicators
        explain_indicators = ["explain", "why", "what is", "how does", "understand", "intuition"]
        if any(indicator in question for indicator in explain_indicators):
            return "explanation"
        
        #mathematical derivation indicators
        math_indicators = ["derive", "show", "proof", "mathematical", "equation"]
        if any(indicator in question for indicator in math_indicators):
            return "mathematical"
        
        #course indicators (for now)
        course_indicators = ["electrostatics", "gauss", "coulomb", "potential"]
        if any(indicator in question for indicator in course_indicators):
            return "course electrostatics"
        
        return "general"

    def _apply_source_filtering(self, chunks, question_type):
        """Filter and prioritize chunks based on question type"""
        
        # Separate chunks by source type
        exercise_chunks = [c for c in chunks if c.metadata.get('source_type') == 'exercise']
        feynman_chunks = [c for c in chunks if c.metadata.get('source_type') == 'feynman']
        course_chunks = [c for c in chunks if c.metadata.get('source_type') not in ['exercise', 'feynman']]
        
        if question_type == "problem_solving":
            # Prioritize exercise materials for problem solving
            return exercise_chunks[:5] + feynman_chunks[:1] + course_chunks[:1]
        
        elif question_type == "explanation":
            # Prioritize Feynman materials for conceptual explanations
            return feynman_chunks[:4]  + exercise_chunks[:2] + course_chunks[:1]
        
        elif question_type == "mathematical":
            # Balance between course materials and exercises for mathematical content
            return  exercise_chunks[:4] + feynman_chunks[:2] + course_chunks[:1]
        
        elif question_type == "course electrostatics":
            return course_chunks[:5] + feynman_chunks[:2]

        else:  # general
            return feynman_chunks[:3] + exercise_chunks[:3] + course_chunks[:1]

    #math heavy filter
    def _is_math_heavy_concept(self, question):
        """Identify concepts that benefit from mathematical treatment"""
        math_heavy_concepts = [
            "maxwell equations", "wave equation", "poynting vector", "boundary conditions",
            "laplacian", "divergence", "curl", "gradient", "differential equation",
            "fourier", "eigenvalue", "complex impedance", "transmission line",
            "waveguide", "antenna theory", "scattering"
        ]
        
        question_lower = question.lower()
        return any(concept in question_lower for concept in math_heavy_concepts)

    def _boost_mathematical_chunks(self, chunks):
        """Prioritize chunks with mathematical content for math-heavy concepts"""
        math_indicators = ["equation", "derivative", "integral", "∇", "∂", "∫", "=", "formula", "theorem"]
        
        math_chunks = []
        non_math_chunks = []
        
        for chunk in chunks:
            content = chunk.page_content
            math_score = sum(1 for indicator in math_indicators if indicator in content)
            
            if math_score >= 2:  # Contains multiple mathematical indicators
                math_chunks.append(chunk)
            else:
                non_math_chunks.append(chunk)
        
        # Return math-heavy chunks first, then conceptual ones
        return math_chunks[:4] + non_math_chunks[:3]

    
    #relevace sort
    def _sort_chunks_by_relevance(self, chunks, question):
        """Sort chunks by comprehensive relevance scoring"""
        scored_chunks = []
        
        for chunk in chunks:
            score = self._calculate_comprehensive_relevance(chunk, question)
            scored_chunks.append((chunk, score))
        
        # Sort by score (highest first) and return just the chunks
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in scored_chunks]

    def _calculate_comprehensive_relevance(self, chunk, question):
        """Calculate detailed relevance score for a chunk"""
        score = 0
        content = chunk.page_content.lower()

        
        #keyword overlap 
        question_words = question.split()
        content_words = set(content.split())
        
        important_words = [w for w in question_words if len(w) > 3 and w not in 
                        ['what', 'how', 'why', 'when', 'where', 'does', 'can', 'will']]
        
        for word in important_words:
            if word in content_words:
                score += 2
                # Bonus for exact phrase matches
                if word in question and word in content:
                    score += 3
        
        
        #electromagnetic concept density
        em_core_terms = ["electric field", "magnetic field", "maxwell", "faraday", "gauss", "ampere", 
                        "electromagnetic", "induction", "flux", "potential", "current", "charge"]
        concept_count = sum(1 for term in em_core_terms if term in content)
        score += concept_count * 2
        
        #content quality indicators
        content_length = len(chunk.page_content)
        if 200 <= content_length <= 1500:  # Optimal length range
            score += 5
        elif content_length < 100:
            score -= 10
        elif content_length > 2000:
            score -= 3
        
        # mathematical formula presence (for technical questions)
        if any(math_word in question for math_word in ["derive", "equation", "formula", "calculate", "prove", "exercise"]):
            formula_indicators = ["=", "∇", "∂", "∫", "∑", "→", "×", "·"]
            formula_count = sum(2 for indicator in formula_indicators if indicator in content)
            score += formula_count * 1.5
        

        
        #penalize very generic or repetitive content
        unique_words = len(set(content.split()))
        total_words = len(content.split())
        if total_words > 0:
            uniqueness_ratio = unique_words / total_words
            if uniqueness_ratio < 0.3: 
                score -= 5
        
        return score
    
    def _smart_search(self, question):
        """Multi-stage search with context-aware filtering"""
        question = question.lower()
        #expand query
        expanded_question = self.expand_electromagnetics_query(question)
        initial_chunks = self.vector_store.search(expanded_question, k=12)
        
        #question type
        question_type = self._detect_question_type(question)
        filtered_chunks = self._apply_source_filtering(initial_chunks, question_type)
        
        #enhance math is math heavy concepts
        if self._is_math_heavy_concept(question):
            math_chunks = self._boost_mathematical_chunks(filtered_chunks)
            filtered_chunks = math_chunks
        
        #lenght validation
        final_chunks = self._sort_chunks_by_relevance(filtered_chunks, question)
        
        return final_chunks[:4] 
    
    # =====MAIN QUESTION METHODS=====
    
    def _build_conversation_context(self, current_question):
        """Build relevant context from conversation history"""
        if not self.conversation_history:
            return ""
        
        #simple relevance check (can be improved)
        current_words = set(current_question.lower().split())
        relevant_exchanges = []
        
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            previous_words = set(exchange['question'].lower().split())
            if len(current_words.intersection(previous_words)) >= 2:
                relevant_exchanges.append(f"Previous Q: {exchange['question']}\nPrevious A: {exchange['answer'][:200]}...")
        
        return "\n".join(relevant_exchanges) if relevant_exchanges else ""

    def _update_conversation_history(self, question, answer, sources):
        """Store conversation for context building"""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "sources": [chunk.metadata['source_file'] for chunk in sources]
        })
        
        #only 10 questions at the time (memory constraint)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def ask_question(self, question, teaching_style="feynman"):
        """Main question-answering with conversation awareness"""
        

        start_time = time.time()
    
        relevant_chunks = self._smart_search(question)
        if not relevant_chunks:
            return "I couldn't find relevant information in your documents. Try rephrasing your question or being more specific."
        
        
        #build context
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            source_type = chunk.metadata.get('source_type', 'unknown')
            source_file = chunk.metadata.get('source_file', 'unknown')
            context_parts.append(f"[Source {i+1} - {source_type.upper()}] {source_file}:\n{chunk.page_content}\n")
        
        context = "\n" + "="*50 + "\n".join(context_parts)
        

        
        #create messages
        messages = [
            SystemMessage(content=self.get_system_prompt(teaching_style))
        ]
        
        #add conversation history if relevant
        conversation_context = self._build_conversation_context(question)
        if conversation_context:
            messages.append(SystemMessage(content=f"Previous conversation context:\n{conversation_context}"))
        
        #main query
        messages.append(HumanMessage(content=f"""Question: {question}
        Relevant context from your documents:{context}
        Provide a clear, educational response that builds genuine understanding. Cite which sources you're using and explain why they're relevant."""))
        
        # Get response
        response = self.llm.invoke(messages)
        
        # Store in conversation history
        self._update_conversation_history(question, response.content, relevant_chunks)

        elapsed_time = time.time() - start_time
        print(f"Response generated in {elapsed_time:.2f} seconds")
        
        return response.content

# Test the tutor
if __name__ == "__main__":
    tutor = ElectromagnetismTutor()
    
    # Test basic functionality
    test_question = "can you explain how the electric field responds in relation with the number of dimensions where uniform charges expand to unity"
    answer = tutor.ask_question(test_question)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")