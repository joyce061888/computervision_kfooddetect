# Korean Dish Classification

## Idea
Create a model that recognizes traditional Korean dishes (bibimbap, tteokbokki, kimchi variations) and provides historical/cultural context.

### Two-level classification:
1. **Level 1 (coarse)**
   - Classify the food group: *Jeongol, Jjigae, Tang, etc.*

2. **Level 2 (fine)**
   - Classify the specific dish under that group:  
     e.g., *Kimchi Jjigae, Doenjang Jjigae, etc.*

## How to run
```bash
python3 train.py
python3 predict.py
