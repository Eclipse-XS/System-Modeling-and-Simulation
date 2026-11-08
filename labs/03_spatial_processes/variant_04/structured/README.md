# Лабораторна робота №3 — Варіант 04 (Структурована ООП-версія)

**Файл:** `src/lab3_structure.py`  
**Походження:** Корінь `SS1\lab3_structure.py`  
**Особливості:**
- Класи `@dataclass HeatProblem` та `HeatSolution`.
- Використання матричного оператора згасання гармонік `decay *= q` замість повторного обчислення експонент на кожному кроці.
- Виділені статичні буфери для коефіцієнтів РК4 (`k1, k2, k3, k4, tmp`).

**Запуск:**
```powershell
& .venv\Scripts\python.exe labs\03_spatial_processes\variant_04\structured\src\lab3_structure.py
```
