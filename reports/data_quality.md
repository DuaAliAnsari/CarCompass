# Data-quality report

Source: `data/raw/pakistan_used_cars.csv`

## Rows after each step

| Step | Before | After | Removed |
|---|---|---|---|
| raw rows | | 61,919 | |
| dropped: missing required field | 61,919 | 61,063 | 856 |
| dropped: price out of range | 61,063 | 61,056 | 7 |
| dropped: year out of range | 61,056 | 60,917 | 139 |
| dropped: mileage out of range | 60,917 | 60,516 | 401 |
| dropped: engine_cc out of range | 60,516 | 60,454 | 62 |
| dropped: old car with implausibly low mileage | 60,454 | 59,225 | 1,229 |
| dropped: duplicate Ad Reference | 59,225 | 59,217 | 8 |
| dropped: exact duplicate listing | 59,217 | 58,405 | 812 |
| dropped: rare make/model | 58,405 | 57,991 | 414 |

Final rows: 57,991 (train 46,335, test 11,656)

Train file hash (sha256, first 16 chars): `af36e9c3f0269c41`

## Era counts

Pre-2010: 17,737 | 2010 and later: 40,254

## Missing values (fraction)

|              |   missing |
|:-------------|----------:|
| engine_cc    |     0.007 |
| body_type    |     0.109 |
| color        |     0     |
| location     |     0     |
| fuel         |     0     |
| transmission |     0     |
| assembly     |     0     |

## Price (PKR)

|       |            price |
|:------|-----------------:|
| count |  57991           |
| mean  |      4.08092e+06 |
| std   |      6.03286e+06 |
| min   | 130000           |
| 25%   |      1.5e+06     |
| 50%   |      2.75e+06    |
| 75%   |      4.5e+06     |
| max   |      1.5e+08     |

## Top 15 make/model

|                       |   listings |
|:----------------------|-----------:|
| ('Toyota', 'Corolla') |       6782 |
| ('Honda', 'Civic')    |       4953 |
| ('Suzuki', 'Mehran')  |       4324 |
| ('Suzuki', 'Alto')    |       4214 |
| ('Honda', 'City')     |       3346 |
| ('Suzuki', 'Cultus')  |       3278 |
| ('Suzuki', 'Wagon R') |       1522 |
| ('Toyota', 'Vitz')    |       1405 |
| ('Daihatsu', 'Mira')  |       1109 |
| ('Suzuki', 'Swift')   |       1023 |
| ('Suzuki', 'Bolan')   |       1016 |
| ('Toyota', 'Yaris')   |        951 |
| ('Toyota', 'Raize')   |        817 |
| ('Toyota', 'Passo')   |        809 |
| ('Toyota', 'Prado')   |        779 |