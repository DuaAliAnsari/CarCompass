import pandas as pd

from src.prepare_multi import assign_splits, drop_cross_source_duplicates, prepare_multi
from tests.test_prepare_combined import B_MAP, make_a, make_b

LOOSE = {"min_model_count": 0, "min_city_count": 0, "check_low_mileage": False}


def test_b_copy_of_a_listing_is_dropped_and_a_copy_kept():
    car = {"make": "Toyota", "model": "Corolla", "year": 2016, "mileage": 75000,
           "city": "Lahore"}
    a = pd.DataFrame([{**car, "price": 3_000_000, "source": "A"}])
    b = pd.DataFrame([{**car, "price": 2_100_000, "source": "B"},
                      {**car, "mileage": 90000, "price": 2_000_000, "source": "B"}])
    a, b, dropped = drop_cross_source_duplicates(a, b)
    assert dropped == 1
    assert b["mileage"].tolist() == [90000]


def test_splits_keep_every_listing_group_on_one_side():
    df, _ = prepare_multi(make_a(600), {}, make_b(600), B_MAP, **LOOSE)
    sides = df.groupby("listing_group")["split"].nunique()
    assert (sides == 1).all()
    assert set(df["split"]) == {"fit", "val", "cal", "test"}
    assert set(df["source"]) == {"A", "B"}


def test_split_shares_are_close_to_the_targets():
    df = pd.DataFrame({"listing_group": [str(i) for i in range(2000)]})
    share = assign_splits(df).value_counts(normalize=True)
    assert abs(share["test"] - 0.2) < 0.02
    assert abs(share["val"] - 0.08) < 0.02
    assert abs(share["cal"] - 0.08) < 0.02


def test_b_prices_are_scaled_and_report_counts_add_up():
    df, report = prepare_multi(make_a(600), {}, make_b(600), B_MAP, **LOOSE)
    assert report["b_price_factor"] > 1  # B's older prices are scaled up
    total = sum(n for split in report["rows_by_split"].values() for n in split.values())
    assert total == report["rows_out"] == len(df)
