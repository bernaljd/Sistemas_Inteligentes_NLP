import pandas as pd
from sklearn.model_selection import train_test_split

ETIQUETA = "polaridad_basica"

df = pd.read_csv("feedback_preprocesado.csv", sep=";")
print("Total de comentarios:", len(df))  # 618

# Algunos comentarios quedaron sin etiqueta ni texto_limpio en el preprocesado.
# stratify no acepta valores vacios, asi que se avisa y se excluyen.
sin_etiqueta = df[df[ETIQUETA].isna() | df["texto_limpio"].isna()]
if len(sin_etiqueta) > 0:
    print(f"\nAVISO: {len(sin_etiqueta)} comentarios sin etiqueta o sin texto_limpio "
          f"(ids: {sin_etiqueta['id'].tolist()}). Se excluyen del split.")
    print("Etiquetalos en feedback_preprocesado.csv para usar los 618 completos.\n")
    df = df.drop(sin_etiqueta.index).reset_index(drop=True)

# Split 80/20 estratificado: train y test mantienen la misma proporcion
# de positivo / neutral / negativo.
train, test = train_test_split(
    df,
    test_size=0.2,
    stratify=df[ETIQUETA],
    random_state=42,  # misma division cada vez que se ejecuta
)

# Verificar que ningun comentario aparece en ambos grupos
assert not set(train["id"]) & set(test["id"]), "Hay ids repetidos entre train y test"
assert not set(train["feedback"]) & set(test["feedback"]), "Hay comentarios repetidos entre train y test"

print(f"Train: {len(train)} comentarios ({len(train) / len(df):.0%})")
print(f"Test:  {len(test)} comentarios ({len(test) / len(df):.0%})")

print("\nProporcion de clases en train:")
print(train[ETIQUETA].value_counts(normalize=True).round(3))
print("\nProporcion de clases en test:")
print(test[ETIQUETA].value_counts(normalize=True).round(3))

# Guardar. test.csv NO se usa hasta el paso 7 (evaluacion final).
train.to_csv("train.csv", index=False, sep=";")
test.to_csv("test.csv", index=False, sep=";")
print("\nGuardados: train.csv y test.csv")
