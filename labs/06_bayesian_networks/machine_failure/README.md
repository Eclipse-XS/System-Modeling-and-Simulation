# Лабораторна робота №6 — Модель відмов обладнання (Канонічна версія)

**Канонічний статус:** 🏆 РЕКОМЕНДОВАНО ДЛЯ ЗАХИСТУ  
**Походження:** Каталог `max_ss\lab6.py`  
**Точка входу:** `src/lab6.py`  
**Датасет:** `data/machine_failure.csv` (10 000 рядків, AI4I 2020 Predictive Maintenance Dataset)  
**Згенерований артефакт:** `outputs/bayesian_network_structure_machine_failure.png`  
**Залежності:** `pandas`, `tabulate`, `pgmpy`, `matplotlib`, `networkx`

---

## 1. Структура графа та змінні

Мережа моделює ймовірність технічної відмови промислового верстата (`Machine failure`) на основі 7 вузлів:
1. `Type` (L, M, H — низький, середній, високий клас якості обладнання);
2. `Air_temp_cat` (температура повітря: low, medium, high);
3. `Process_temp_cat` (температура процесу обробки: low, medium, high);
4. `Speed_cat` (швидкість обертання шпинделя, об/хв: low, medium, high);
5. `Torque_cat` (крутний момент, Нм: low, medium, high);
6. `Wear_cat` (час зносу різця, хв: low, medium, high);
7. `Machine failure` (цільова змінна: no / yes).

### Орієнтовані ребра графа:
- `Type` $\to$ `Torque_cat`
- `Type` $\to$ `Speed_cat`
- `Air_temp_cat` $\to$ `Process_temp_cat`
- `Torque_cat` $\to$ `Machine failure`
- `Wear_cat` $\to$ `Machine failure`
- `Speed_cat` $\to$ `Machine failure`
- `Process_temp_cat` $\to$ `Machine failure`

Граф є строго ациклічним (DAG) і задовольняє `model.check_model() == True`.

---

## 2. Верифіковані результати імовірнісного виведення

| № | Назва сценарію | Вхідні свідоцтва (Evidence) | Найбільш імовірний стан | $P(\text{failure}=\text{yes})$ |
|---|---|---|---|---|
| 1 | Апріорна ймовірність (базова) | *Свідоцтва відсутні* | no ($91.33\%$) | **$8.67\%$** |
| 2 | Критичне перевантаження | Високий знос (`high`) + високий момент (`high`) | no ($76.63\%$) | **$23.37\%$** (ризик зріс у 2.7 раза) |
| 3 | Сприятливі умови | Низький знос (`low`) + низький момент (`low`) | no ($82.84\%$) | **$17.16\%$** |
| 4 | Теплове перевантаження + швидкість | Висока temp процесу + високий знос + висока швидкість | no ($82.87\%$) | **$17.13\%$** |
| 5 | Високий клас обладнання під навантаженням | Type = H + високий момент + високий знос | no ($76.82\%$) | **$23.18\%$** |
| 6 | Дослідження d-сепарації | Температура повітря = висока | $P(\text{Process\_temp}=\text{high}) = \mathbf{70.08\%}$ | *Проміжний вузол* |

---

## 3. Інструкція із запуску

```powershell
# Запуск із кореня проєкту:
& .venv\Scripts\python.exe labs\06_bayesian_networks\machine_failure\src\lab6.py

# Або запуск безпосередньо з каталогу моделі:
cd labs\06_bayesian_networks\machine_failure
& ..\..\..\.venv\Scripts\python.exe src\lab6.py
```
*(Скрипт роздрукує повний паспорт даних, таблиці CPD у форматі `tabulate`, результати всіх 6 виведень та збереже зображення графа в `outputs/`).*
