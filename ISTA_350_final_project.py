"""
AUTHOR: Bolor-Erdene Altanmunkh
DATE: Apr 27, 2026
CLASS: ISTA 350
SECTION LEADER: Jacob Jaeger
DESCRIPTION: Final project. 
COLLABORATORS: None 

OUTPUTS:
1. car_sales_top10.png
2. car_sales_duration_vs_sales.png
3. car_sales_manufacturer_avg.png
"""


import re
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


URL = "https://en.wikipedia.org/wiki/List_of_automobile_sales_by_model"


def clean_sales(value):
    """Extract the sales figure from the Sales column, preferring comma formatted numbers."""
    if pd.isna(value):
        return np.nan

    text = str(value)
    match = re.search(r"\d{1,3}(?:,\d{3})+", text)
    if match:
        return int(match.group().replace(",", ""))

    plain = re.search(r"\d+", text)
    if plain:
        return int(plain.group())
    return np.nan


def clean_production_years(value):
    """Estimate production duration from the Production column."""
    if pd.isna(value):
        return np.nan

    text = str(value)

    years = re.findall(r"\d{4}", text)

    if len(years) >= 2:
        start = int(years[0])
        end = int(years[1])
        return end - start + 1

    if len(years) == 1 and "present" in text.lower():
        start = int(years[0])
        return 2026 - start + 1

    return np.nan


def scrape_car_sales_data():
    """Scrape and combine usable tables from Wikipedia."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ISTA350-project/1.0)"}
    tables = pd.read_html(URL, storage_options={"User-Agent": headers["User-Agent"]})

    cleaned_tables = []

    for table in tables:
        if "Automobile" in table.columns and "Sales" in table.columns:
            temp = table.copy()

            if "Manufacturer" not in temp.columns:
                temp["Manufacturer"] = (temp["Automobile"].astype(str).str.extract(r"^(\S+)", expand=False).fillna("Unknown"))

            if "Production" not in temp.columns:
                temp["Production"] = np.nan

            temp = temp[["Manufacturer", "Automobile", "Production", "Sales"]]
            cleaned_tables.append(temp)

    df = pd.concat(cleaned_tables, ignore_index=True)

    df["Sales_clean"] = df["Sales"].apply(clean_sales)
    df["Production_years"] = df["Production"].apply(clean_production_years)


    df = df.dropna(subset=["Sales_clean"])
    df = df[df["Sales_clean"] > 0]

    return df


def plot_top_10_sales(df):
    """Bar chart: Top 10 best selling car models."""
    top10 = df.sort_values("Sales_clean", ascending=False).head(10)

    plt.figure(figsize=(12, 7))
    plt.barh(top10["Automobile"], top10["Sales_clean"])
    plt.xlabel("Total Sales (vehicles)")
    plt.ylabel("Automobile Model")
    plt.title("Top 10 Best selling automobile models ")
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig("car_sales_top10.png", dpi=300)
    plt.show()


def plot_duration_vs_sales(df):
    """Scatter plot with regression line and R^2 and p-value analysis."""
    plot_df = df.dropna(subset=["Production_years", "Sales_clean"])

    x = plot_df["Production_years"]
    y = plot_df["Sales_clean"]

    # Statsmodels regression
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()

    intercept = model.params["const"]
    slope = model.params["Production_years"]
    r_squared = model.rsquared
    p_value = model.pvalues["Production_years"]

    regression_line = intercept + slope * x

    print(f"R-squared: {r_squared:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("Interpretation: The relationship is statistically significant at the 0.05 level.")
    else:
        print("Interpretation: The relationship is not statistically significant at the 0.05 level.")

    plt.figure(figsize=(10, 7))
    plt.scatter(x, y, alpha=0.6)
    plt.plot(x, regression_line)

    plt.xlabel("Production Duration (years)")
    plt.ylabel("Total Sales (vehicles)")
    plt.title(
        f"Production Duration vs Automobile Sales\n"
        f"R² = {r_squared:.3f}, p = {p_value:.4f}"
    )

    plt.gca().yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig("car_sales_duration_vs_sales.png", dpi=300)
    plt.show()


def plot_manufacturer_average_sales(df):
    """Bar chart: Average sales by manufacturer."""
    manufacturer_sales = (
        df.groupby("Manufacturer")["Sales_clean"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )

    plt.figure(figsize=(12, 7))
    plt.barh(manufacturer_sales.index, manufacturer_sales.values)
    plt.xlabel("Average sales per model (number of vehicles)")
    plt.ylabel("Manufacturer")
    plt.title("Top 10 Manufacturers by Average sales per model")
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    plt.tight_layout()
    plt.savefig("car_sales_manufacturer_avg.png", dpi=300)
    plt.show()


def main():
    df = scrape_car_sales_data()

    print("Scraped rows:", len(df))
    print(df.head())

    df.to_csv("cleaned_car_sales_data.csv", index=False)

    plot_top_10_sales(df)
    plot_duration_vs_sales(df)
    plot_manufacturer_average_sales(df)


if __name__ == "__main__":
    main()