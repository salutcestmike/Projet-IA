from pathlib import Path
import imodmodel

# Fichier source
mod_file = Path("./data/HCT116-0Gy-3j-2900X-0002.mod")
model = imodmodel.ImodModel.from_file(mod_file)

for i, obj in enumerate(model.objects):
    print(
        f"Objet {i} : "
        f"{obj.header.name!r} — "
        f"{len(obj.contours)} contours"
    )
