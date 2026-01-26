from ultralytics import YOLO

model_path = "best.pt"   # 모델 경로
model = YOLO(model_path)

print("=== YOLO Model Classes ===")
print(model.names)

print("\n--- Class List ---")
for k, v in model.names.items():
    print(f"{k}: {v}")


'''
=== YOLO Model Classes ===
{0: 'Fanta', 1: 'Coke', 2: 'Coke_Light', 3: 'Pepsi', 4: 'Pepsi_Max'}

--- Class List ---
0: Fanta
1: Coke
2: Coke_Light
3: Pepsi
4: Pepsi_Max


'''