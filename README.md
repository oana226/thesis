from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
mech = "stiff"
geom = "abs_curvature"

area = "S_front_inf"

rows = []

for sub in subject_data_on_fsaverage.keys():

    x_all = []
    y_all = []

    for hemi in ["lh", "rh"]:

        region_idx = parcel_labels[area][hemi]

        mask = labels[hemi][0] == region_idx

        geom_data = subject_data_on_fsaverage[sub][geom][hemi]
        mech_data = subject_data_on_fsaverage[sub][mech][hemi]

        if geom_data is None or mech_data is None:
            continue

    
        x = geom_data[mask]
        y = mech_data[mask]

        valid = np.isfinite(x) & np.isfinite(y)

        x_all.extend(x[valid])
        y_all.extend(y[valid])

    # store all vertices from this subject
    for xv, yv in zip(x_all, y_all):

        rows.append({
            "subject": sub,
            geom: xv,
            mech: yv
        })

df = pd.DataFrame(rows)

subjects = df["subject"].unique()

train_subjects, test_subjects = train_test_split(subjects, test_size=0.2, random_state=42)

train_df = df[df["subject"].isin(train_subjects)]
test_df  = df[df["subject"].isin(test_subjects)]

X_train = train_df[[geom]]
y_train = train_df[mech]

X_test = test_df[[geom]]
y_test = test_df[mech]

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R²: {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

X_train_sm = sm.add_constant(X_train)

ols_model = sm.OLS(y_train, X_train_sm).fit()

print(ols_model.summary())

