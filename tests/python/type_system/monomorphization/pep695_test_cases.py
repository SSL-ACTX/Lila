from lirien import verify, i64, f64
from tests.python.type_system.monomorphization.test_generic_data import BoxedVal, Opt


@verify
def get_boxed_pep695[U](b: BoxedVal[U]) -> U:
    return b.value


@verify
def unwrap_opt_pep695[U](o: Opt[U], default_val: U) -> U:
    match o:
        case Opt_U.Some(val):
            return val
        case Opt_U.None_:
            return default_val
