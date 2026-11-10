# Data source and file mapping

This project uses the two CSVs supplied by the project author as `Customer Churn dataset(1).zip`:

- `train.csv` = original `customer_churn_dataset-training-master.csv` (440,833 rows before removing a single entirely blank row)
- `test.csv` = original `customer_churn_dataset-testing-master.csv` (64,374 rows)

The dataset is publicly available at:

- https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset

A publicly viewable mirror of the training CSV also exists at:

- https://huggingface.co/datasets/MatanKriel/Matan-Kriel-Customer-Churn-dataset

These references help users identify the dataset; they do not establish a verified chain of ownership or redistribution license. Check the dataset provider's current terms before publishing raw files in a public GitHub repository.

`CustomerID` is never used for modeling. The chosen data do **not** have monthly charges, total charges, or internet-service fields; see `app/schemas.py` for the correct API contract.
