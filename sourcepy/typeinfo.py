# IDL-style usage
type_array, error = typeidx('Ia-91T')
if error == 0:
    print("Found match")

# Object-oriented usage
info = SNTypeInfo()
matches = info.find_type_by_name('II')
for it, ist in matches:
    print(f"Type {it}, Subtype {ist}: {info.get_type_name(it, ist)}")

# Get all Ia subtypes
ia_types = [info.get_type_name(1, ist) for ist in range(1, 7)]
