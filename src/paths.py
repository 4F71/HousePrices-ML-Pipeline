from pathlib import Path  # 'Path' büyük harfle olmalı



ROOT = Path(__file__).resolve().parents[1]  # proje kök dizinini bul
DATA_DIR = ROOT / "data" #veri klasörü
MODELS_DIR = ROOT / "models" # modelin kaydedileceği
REPORTS_DIR  = ROOT / "reports" #metrik raporlarının kaydedileceği klasör
FIGURES_DIR  = ROOT / "figures" #grafik kaydedilen klasör


if __name__  == "__main__":

    print("ROOT",ROOT)
    print("DATA_DIR",DATA_DIR)
    print("MODELS_DIR",MODELS_DIR)
    print("REPORTS_DIR",REPORTS_DIR)
    print("FIGURES_DIR",FIGURES_DIR)
