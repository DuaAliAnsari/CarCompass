import numpy as np

from src.prepare_data import (
    GROUP_COLS,
    drop_duplicates,
    map_schema,
    parse_number,
    parse_price_pkr,
    prepare,
    validate,
)


def test_parse_price_handles_lacs_crore_and_plain_numbers():
    assert parse_price_pkr("27.4 lacs") == 2_740_000
    assert parse_price_pkr("1.2 crore") == 12_000_000
    assert parse_price_pkr("2,740,000") == 2_740_000
    assert parse_price_pkr(1_500_000) == 1_500_000
    assert np.isnan(parse_price_pkr("call for price"))
    assert np.isnan(parse_price_pkr(None))


def test_parse_number_extracts_digits():
    assert parse_number("75,000 km") == 75_000
    assert parse_number("1300 cc") == 1300
    assert np.isnan(parse_number("n/a"))


def test_validate_drops_missing_and_impossible_rows(raw_sample, column_map):
    df = map_schema(raw_sample, column_map)
    clean, report = validate(df)
    assert report["dropped_missing"] >= 1       # the row with no price
    assert report["dropped_out_of_range"] >= 2  # year 1850 and the absurd price
    assert clean["year"].between(1980, 2100).all()
    assert clean["price"].between(50_000, 200_000_000).all()
    assert clean[["make", "model", "year", "mileage", "price"]].notna().all().all()


def test_drop_duplicates_removes_exact_copy(raw_sample, column_map):
    clean, _ = validate(map_schema(raw_sample, column_map))
    deduped, dropped = drop_duplicates(clean)
    assert dropped >= 1
    assert len(deduped) == len(clean) - dropped


def test_split_is_reproducible_and_has_no_group_leakage(raw_sample, column_map):
    train1, test1, report = prepare(raw_sample, column_map, seed=42)
    train2, test2, _ = prepare(raw_sample, column_map, seed=42)
    assert train1.equals(train2) and test1.equals(test2)
    assert report["train_rows"] + report["test_rows"] == report["rows_out"]

    def keys(df):
        return set(df[GROUP_COLS].astype(str).agg("|".join, axis=1))

    assert keys(train1).isdisjoint(keys(test1))
    assert 0.1 < len(test1) / report["rows_out"] < 0.35
