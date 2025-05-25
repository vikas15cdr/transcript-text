import argparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
from sentence_transformers import SentenceTransformer
from groq import Groq
import os
from dotenv import load_dotenv

class TranscriptProcessor:
    def __init__(self, transcript_file):
        self.transcript_file = transcript_file
        self.chunks = []
        self.timestamps = []
        self._parse_transcript()
        
    def _parse_transcript(self):
        """Parse the transcript file into chunks with timestamps"""
        with open(self.transcript_file, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Split by timestamp pattern
        pattern = re.compile(r'(\[\d{2}:\d{2} - \d{2}:\d{2}\]) (.*?)(?=\[\d{2}:\d{2} - \d{2}:\d{2}\]|\Z)', re.DOTALL)
        matches = pattern.finditer(content)
        
        for match in matches:
            timestamp = match.group(1)
            text = match.group(2).strip()
            if text:  # Only add if there's actual text
                self.timestamps.append(timestamp)
                self.chunks.append(text)
    
    def get_chunks(self):
        """Return chunks and timestamps"""
        return self.chunks, self.timestamps

class SemanticSearch:
    def __init__(self, chunks, timestamps):
        self.chunks = chunks
        self.timestamps = timestamps
    
    def tfidf_search(self, question, top_k=1):
        """Search using TF-IDF and cosine similarity"""
        vectorizer = TfidfVectorizer()
        chunk_vectors = vectorizer.fit_transform(self.chunks)
        question_vector = vectorizer.transform([question])
        
        similarities = cosine_similarity(question_vector, chunk_vectors)
        top_indices = np.argsort(similarities[0])[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((self.timestamps[idx], self.chunks[idx], similarities[0][idx]))
        return results
    
    def groq_llm_search(self, question, top_k=1):
        """Search using GROQ API with Mixtral model"""
        try:
            load_dotenv()
            client = Groq(api_key=os.getenv("groq_key"))
            
            model = SentenceTransformer('all-MiniLM-L6-v2')
            chunk_embeddings = model.encode(self.chunks)
            question_embedding = model.encode(question)
            
            similarities = cosine_similarity([question_embedding], chunk_embeddings)
            top_indices = np.argsort(similarities[0])[-top_k:][::-1]
            \
            context = "\n\n".join([f"{self.timestamps[i]}: {self.chunks[i]}" for i in top_indices])
            
            response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided transcript context. Only use information from the context. If you don't know the answer, say so."
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=1024
            )
            
            answer = response.choices[0].message.content
            best_chunk_idx = top_indices[0]
            return [(self.timestamps[best_chunk_idx], answer, similarities[0][best_chunk_idx])]
            
        except Exception as e:
            print(f"Error with GROQ search: {e}")
            return []
        
    def openai_search(self, question, top_k=1):
        """Search using OpenAI embeddings"""
        try:
            client = openai.OpenAI()
            

            chunk_embeddings = []
            for chunk in self.chunks:
                response = client.embeddings.create(
                    input=chunk,
                    model="text-embedding-3-small"
                )
                chunk_embeddings.append(response.data[0].embedding)
            
  
            response = client.embeddings.create(
                input=question,
                model="text-embedding-3-small"
            )
            question_embedding = response.data[0].embedding
            
            # Calculate similarities
            similarities = cosine_similarity([question_embedding], chunk_embeddings)
            top_indices = np.argsort(similarities[0])[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append((self.timestamps[idx], self.chunks[idx], similarities[0][idx]))
            return results
            
        except Exception as e:
            print(f"Error with OpenAI search: {e}")
            return []
    

def main():
    parser = argparse.ArgumentParser(description="Semantic Search for Transcript Q&A")
    parser.add_argument("transcript_file", help="Path to the transcript text file")
    parser.add_argument("method", choices=['tfidf', 'openai', 'groq'], 
                       help="Search method: tfidf, openai, or groq")
    args = parser.parse_args()
    
    # Process transcript
    processor = TranscriptProcessor(args.transcript_file)
    chunks, timestamps = processor.get_chunks()
    search = SemanticSearch(chunks, timestamps)
    
    print("Transcript loaded, please ask your question (press 8 for exit):")
    
    while True:
        question = input("> ")
        if question == '8':
            break
        
        if args.method == 'tfidf':
            results = search.tfidf_search(question)
        elif args.method == 'openai':
            results = search.openai_search(question)
        elif args.method == 'groq':
            results = search.groq_llm_search(question)
        
        if results:
            for timestamp, chunk, score in results:
                print(f"\n[{timestamp}], {chunk}\n(Relevance score: {score:.3f})")
        else:
            print("No results found or error occurred.")
        
        print("\nAsk another question or press 8 to exit:")

if __name__ == "__main__":
    main()