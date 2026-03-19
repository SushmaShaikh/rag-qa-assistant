from rag_pipeline import answer_question

def main():
    print("Combined RAG QA Assistant (CLI)")
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nAnswer:", answer_question(q))

if __name__ == "__main__":
    main()