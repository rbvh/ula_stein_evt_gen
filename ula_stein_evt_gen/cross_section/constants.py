VECTOR_BOSON_MASSES = {
    "w": 80.379,
    "z": 91.1876,
}


def vector_boson_mass(vb):
    try:
        return VECTOR_BOSON_MASSES[vb]
    except KeyError as exc:
        raise ValueError(f"Unsupported vector boson type: {vb}") from exc
