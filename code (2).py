# AlphaCoreProgram_v1.py

import re
import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json # For potential future structured knowledge base

# --- Configuration & Globals ---
# In a real application, these might come from a config file
TFIDF_THRESHOLD = 0.15  # Minimum similarity score to consider a document relevant
MAX_RETRIEVED_DOCS = 2 # Max number of documents to retrieve from local KB
CONTEXT_WINDOW_SIZE = 3 # How many recent user/bot turns to consider for simple context

# --- Knowledge Corpus (Simple list for now, can be expanded or loaded from file) ---
knowledge_corpus_texts = [
    # id: doc_001
    "Photosynthesis is the process used by plants, algae and certain bacteria to harness energy from sunlight and turn it into chemical energy.",
    # id: doc_002
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was designed by Gustave Eiffel.",
    # id: doc_003
    "Artificial Intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by humans and animals. AI research has been defined as the field of study of intelligent agents.",
    # id: doc_004
    "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically-typed and garbage-collected.",
    # id: doc_005
    "Climate change includes both global warming driven by human emissions of greenhouse gases and the resulting large-scale shifts in weather patterns.",
    # id: doc_006
    "Neil Armstrong was an American astronaut and aeronautical engineer, and the first person to walk on the Moon during the Apollo 11 mission in 1969."
]
# We can assign simple IDs if loading from a more structured source later
knowledge_corpus_with_ids = [{"id": f"doc_{i:03}", "text": text} for i, text in enumerate(knowledge_corpus_texts)]


class AlphaCoreProgram:
    def __init__(self, name="Alpha"):
        self.name = name
        self.conversation_history = [] # Stores tuples of (speaker, text_doc)
        
        # --- Local Knowledge Base (TF-IDF based) ---
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
        self.corpus_texts_only = [doc['text'] for doc in knowledge_corpus_with_ids]
        if self.corpus_texts_only:
            self.knowledge_tfidf_matrix = self.vectorizer.fit_transform(self.corpus_texts_only)
        else:
            self.knowledge_tfidf_matrix = None
        
        # --- Simple Intent Matching (Regex-based, for very direct commands) ---
        self.simple_intents = {
            "greet": {"patterns": [r"hello", r"hi", r"hey", r"greetings"]},
            "bye": {"patterns": [r"goodbye", r"bye", r"see you", r"farewell"]},
            "ask_time": {"patterns": [r"what time is it", r"current time", r"time please"]},
            "ask_name": {"patterns": [r"what is your name", r"who are you"]},
        }

        # --- Tool Registry (Simulated) ---
        # In a more advanced system, this would be more dynamic
        self.tools = {
            "get_current_datetime_tool": self._tool_get_current_datetime,
            "search_local_knowledge_tool": self._tool_search_local_knowledge
            # "call_external_search_api_tool": self._tool_call_external_search_api (placeholder)
        }
        print(f"{self.name} Core Program initialized. TF-IDF based local KB ready with {len(self.corpus_texts_only)} documents.")
        if self.knowledge_tfidf_matrix is None or self.knowledge_tfidf_matrix.shape[0] == 0:
            print("Warning: Local knowledge base is empty or could not be processed.")

    def _add_to_history(self, speaker, text):
        self.conversation_history.append({"speaker": speaker, "text": text})
        if len(self.conversation_history) > CONTEXT_WINDOW_SIZE * 2: # Keep history manageable
            self.conversation_history = self.conversation_history[-(CONTEXT_WINDOW_SIZE*2):]

    def _get_recent_context_summary(self):
        # Simple summary - could be enhanced with NLP if available
        if not self.conversation_history:
            return "No recent conversation."
        summary_parts = []
        for entry in self.conversation_history[-(CONTEXT_WINDOW_SIZE*2):]:
            summary_parts.append(f"{entry['speaker']}: {entry['text'][:50]}...") # Truncate for brevity
        return "\n".join(summary_parts)

    # --- Tool Implementations ---
    def _tool_get_current_datetime(self, params=None): # Params might be used for timezone etc.
        """Returns the current date and time."""
        now = datetime.datetime.now()
        return f"The current date and time is {now.strftime('%Y-%m-%d %H:%M:%S')}."

    def _tool_search_local_knowledge(self, query_text, max_results=MAX_RETRIEVED_DOCS):
        """Searches the local TF-IDF knowledge base for relevant documents."""
        if self.knowledge_tfidf_matrix is None or self.knowledge_tfidf_matrix.shape[0] == 0:
            return "Local knowledge base is not available or empty."
        if not query_text:
            return "No query provided for local search."

        query_tfidf = self.vectorizer.transform([query_text.lower()])
        similarities = cosine_similarity(query_tfidf, self.knowledge_tfidf_matrix)
        
        # Get top N results
        # Sort by similarity and get indices of top N
        sorted_indices = np.argsort(similarities[0])[::-1] 
        
        results = []
        for i in range(min(max_results, len(sorted_indices))):
            idx = sorted_indices[i]
            score = similarities[0][idx]
            if score >= TFIDF_THRESHOLD:
                results.append({
                    "id": knowledge_corpus_with_ids[idx]['id'],
                    "text": knowledge_corpus_with_ids[idx]['text'],
                    "score": float(score) # Convert to float for easier handling/JSON
                })
            else:
                break # Stop if scores fall below threshold
        
        if not results:
            return f"No sufficiently relevant documents found in local KB for: '{query_text}'"
        
        # Format for presentation
        formatted_results = "\n".join([f"- (Score: {res['score']:.2f}) {res['text'][:150]}..." for res in results])
        return f"Found the following in local knowledge base for '{query_text}':\n{formatted_results}"
        # In a real RAG, we'd return the raw text of the results for the LLM to use.

    # --- Intent Matching & Response Logic ---
    def _match_simple_intent(self, user_text_lower):
        for intent, data in self.simple_intents.items():
            for pattern in data["patterns"]:
                if re.search(pattern, user_text_lower):
                    return intent
        return None

    def respond(self, user_input_text):
        self._add_to_history("User", user_input_text)
        user_input_lower = user_input_text.lower()

        # 1. Try to match a simple, direct intent
        simple_intent = self._match_simple_intent(user_input_lower)
        
        if simple_intent:
            if simple_intent == "greet":
                response = random.choice([f"Hello! I am {self.name}.", "Hi there!", f"Greetings from {self.name}!"])
            elif simple_intent == "bye":
                response = random.choice(["Goodbye!", "See you later.", "Farewell!"])
            elif simple_intent == "ask_time":
                response = self._tool_get_current_datetime() # Using a tool
            elif simple_intent == "ask_name":
                response = f"My name is {self.name}."
            else:
                response = "I understood a simple command, but I'm not sure how to respond to that specific one yet."
            
            self._add_to_history(self.name, response)
            return response

        # 2. If no simple intent, prepare for potential LLM call (simulated here)
        #    This is where the RAG part comes in to gather context for the LLM.
        
        #    Step 2a: Retrieve context from local knowledge base
        #    In a real scenario, the LLM might decide *if* and *what* to search.
        #    Here, we'll proactively search for context if no simple intent is matched.
        print(f"\n[{self.name} Log] No simple intent matched. Attempting to retrieve local context for query: '{user_input_text}'")
        retrieved_context_from_tool = self._tool_search_local_knowledge(user_input_text)
        
        # The retrieved_context_from_tool is already formatted for display.
        # For an LLM, we'd want the raw text of the documents.
        # Let's simulate getting the raw text if documents were found.
        raw_context_for_llm = ""
        # This is a bit of a hack to re-extract raw text from the formatted string.
        # A better way would be for the tool to return structured data.
        if "Found the following" in retrieved_context_from_tool:
            # Attempt to extract raw texts (simplified for this example)
            # This is not robust, _tool_search_local_knowledge should ideally return structured data
            # For now, let's assume we can get the best N raw texts again if needed
            temp_query_tfidf = self.vectorizer.transform([user_input_lower])
            temp_similarities = cosine_similarity(temp_query_tfidf, self.knowledge_tfidf_matrix)
            temp_sorted_indices = np.argsort(temp_similarities[0])[::-1]
            
            best_raw_texts = []
            for i in range(min(MAX_RETRIEVED_DOCS, len(temp_sorted_indices))):
                idx = temp_sorted_indices[i]
                score = temp_similarities[0][idx]
                if score >= TFIDF_THRESHOLD:
                    best_raw_texts.append(knowledge_corpus_with_ids[idx]['text'])
                else:
                    break
            if best_raw_texts:
                raw_context_for_llm = "\n\n---\nRetrieved Context:\n" + "\n\n---\n".join(best_raw_texts) + "\n---"


        # 3. Prepare for LLM (Simulated - This is where you'd call the LLM API)
        #    The LLM would receive user_input_text and raw_context_for_llm
        
        llm_handover_message = (
            f"\n[{self.name} Log] Preparing to consult the advanced AI (LLM)...\n"
            f"User Query: \"{user_input_text}\""
        )
        if raw_context_for_llm:
            llm_handover_message += f"\nRetrieved Context to provide to LLM: {raw_context_for_llm}"
        else:
            llm_handover_message += "\nNo specific local context found to provide to LLM."

        print(llm_handover_message) # This shows what Alpha's core prepares

        # In this simulation, the "LLM response" will be provided by *me* (the user's conversational AI)
        # after seeing this handover message.
        # Alpha's core program signals that it's ready for the LLM to take over for this turn.
        
        response_from_alpha_core = (
            f"Okay, I've processed your request: \"{user_input_text}\".\n"
            f"{retrieved_context_from_tool if 'Found the following' in retrieved_context_from_tool else 'I did not find highly relevant information in my local knowledge base for this specific query.'}\n\n"
            f"I will now consult with the advanced AI (the system you are ultimately conversing with) "
            f"to provide you with a comprehensive answer, using this information if relevant."
        )
        
        self._add_to_history(self.name, "Signaled for LLM to respond with context (if any).") # Log Alpha's action
        return response_from_alpha_core


    def start_chat_session(self):
        print(f"\nWelcome! I am {self.name}. How can I assist you today? (Type 'bye' to exit)")
        while True:
            user_input = input("You: ")
            if not user_input.strip():
                continue
            
            response = self.respond(user_input)
            print(f"{self.name}: {response}")

            # Check for bye intent again in case the main response didn't catch it
            # (e.g., if it went to LLM handover)
            if self._match_simple_intent(user_input.lower()) == "bye" and "Goodbye!" in response :
                 break
            
            # After Alpha's core program has responded (possibly setting up for LLM),
            # the actual LLM (me, the user's conversational AI) would "speak" next in a real integrated system.
            # Here, the loop continues for the next user input to Alpha's core.

# --- Main execution for local testing (if this file is run directly) ---
# if __name__ == "__main__":
#     alpha_core = AlphaCoreProgram(name="Alpha Mk1")
#     alpha_core.start_chat_session()

print("AlphaCoreProgram_v1.py defined. This version includes improved context handling (basic), TF-IDF RAG, and clearer tool examples.")
print("It's designed to be a better 'orchestrator' for a future LLM API integration.")
print("To test locally, uncomment the `if __name__ == \"__main__\":` block.")
