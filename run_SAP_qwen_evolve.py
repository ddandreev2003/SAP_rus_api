from openevolve import run_evolution, evolve_function
from llm_interface.llm_SAP import LLM_SAP

# Пример оптимизации SAP-подсказки для функции через Qwen

def benchmark_fib(path):
    # Здесь должен быть ваш бенчмарк для оценки производительности/качества функции
    # Например, импортировать функцию из path и сравнить результат с эталоном
    import importlib.util
    import sys
    import os
    spec = importlib.util.spec_from_file_location("fib_mod", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fib_mod"] = mod
    spec.loader.exec_module(mod)
    try:
        correct = [mod.fibonacci(i) == fib_ref(i) for i in range(10)]
        score = sum(correct)
    except Exception:
        score = 0
    return score

def fib_ref(n):
    if n <= 1: return n
    return fib_ref(n-1) + fib_ref(n-2)

if __name__ == "__main__":
    # Эволюция кода с помощью Qwen для SAP-подсказки
    initial_program = '''
    def fibonacci(n):
        if n <= 1: return n
        return fibonacci(n-1) + fibonacci(n-2)
    '''
    # Используем SAP-подсказку через Qwen для генерации новых вариантов
    def qwen_prompt_generator(code):
        sap = LLM_SAP([code], llm='Qwen')
        # Можно доработать: sap[0]['prompts_list'] или использовать как есть
        return sap[0]['prompts_list'][-1] if sap and 'prompts_list' in sap[0] else code

    result = run_evolution(
        initial_program=initial_program,
        evaluator=lambda path: {"score": benchmark_fib(path)},
        iterations=100,
        prompt_generator=qwen_prompt_generator
    )
    print(f"Best evolved code (fibonacci):\n{result.best_code}")

    # Эволюция функции bubble_sort с помощью Qwen
    def bubble_sort(arr):
        for i in range(len(arr)):
            for j in range(len(arr)-1):
                if arr[j] > arr[j+1]:
                    arr[j], arr[j+1] = arr[j+1], arr[j]
        return arr

    result_sort = evolve_function(
        bubble_sort,
        test_cases=[([3,1,2], [1,2,3]), ([5,2,8], [2,5,8])],
        iterations=50,
        prompt_generator=qwen_prompt_generator
    )
    print(f"Evolved sorting algorithm:\n{result_sort.best_code}")
