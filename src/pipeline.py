"""
pipeline.py
------------
Preprocessor ve model adımlarını tek bir sklearn Pipeline içinde birleştirir.
"""

from sklearn.pipeline import Pipeline

def build_pipeline(preprocessor, model) -> Pipeline:
    """
    Preprocessor (ön işleme) ve model adımlarını tek bir sklearn Pipeline içinde birleştirir.
    """
    pipe= Pipeline(steps=[
        ("preprocessor",preprocessor), #veri dönüşümü
        ("model",model)               #model uygunalması
    ])
    return pipe

if __name__ == "__main__":
    from src.data import load_data
    from src.preprocess import build_preprocessor
    from src.model import get_model

    df=load_data("train.csv")
    numeric=df.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorial = df.select_dtypes(include=["object"]).columns.tolist()

    pre = build_preprocessor(numeric,categorial)
    model = get_model("ridge")
    pipe = build_pipeline(pre,model)

    print("pipeline adımları",pipe.named_steps.keys())
