from src.generate_data import make_customers
from src.train import TARGET, build_model

def test_synthetic_data_schema_and_reproducibility():
    a = make_customers(1000, 7)
    b = make_customers(1000, 7)
    assert a.equals(b)
    assert TARGET in a.columns
    assert set(a[TARGET].unique()) == {0, 1}
    assert a.customer_id.is_unique

def test_model_pipeline_fits_small_sample():
    frame = make_customers(800, 11)
    X = frame.drop(columns=["customer_id", TARGET])
    y = frame[TARGET]
    model = build_model(frame)
    model.fit(X.iloc[:600], y.iloc[:600])
    probabilities = model.predict_proba(X.iloc[600:])[:, 1]
    assert len(probabilities) == 200
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
