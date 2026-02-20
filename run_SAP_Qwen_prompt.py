from llm_interface.llm_SAP import LLM_SAP

def main():
    description = input("Введите описание задачи для генерации кода по SAP-технике: ")
    result = LLM_SAP([description], llm='Qwen')
    print("\nРезультат генерации по SAP (Qwen):")
    print(result)

if __name__ == "__main__":
    main()
