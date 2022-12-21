from hypothesis import given

# local
import ivy
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_frontend_test


@handle_frontend_test(
    fn_tree="tensorflow.random.uniform",
    dtype_and_min=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=-1000,
        max_value=100,
        min_num_dims=1,
    ),
    dtype_and_max=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        min_value=101,
        max_value=1000,
        min_num_dims=1,
    ),
    dtype_and_seed=helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("float"),
        max_num_dims=1,
    ),
    dtypes=helpers.get_dtypes("float", full=False),
    shape=helpers.get_shape(),
)
def test_tensorflow_random_uniform(
    *,
    dtype_and_min,
    dtype_and_max,
    dtype_and_seed,
    dtypes,
    num_positional_args,
    shape,
    as_variable,
    native_array,
    with_out,
    frontend,
    fn_tree,
    on_device,
):
    min_dtype, minval = dtype_and_min
    max_dtype, maxval = dtype_and_max
    seed_dtype, seed = dtype_and_seed

    # print('oooooooooooooooooooooo')
    # print(f'{minval[0]=}, {type(minval[0])=}')
    # print(f'{maxval[0]=}, {type(maxval[0])=}')
    # print('oooooooooooooooooooooo')

    helpers.test_frontend_function(
        input_dtypes=min_dtype + max_dtype + seed_dtype,
        as_variable_flags=as_variable,
        with_out=False,
        num_positional_args=num_positional_args,
        native_array_flags=native_array,
        frontend=frontend,
        fn_tree=fn_tree,
        on_device=on_device,
        shape=shape,
        minval=minval[0],
        maxval=minval[0],
        seed=30,
        dtype=ivy.float64,  # Right way AFAIK, but breaks with an error
        # dtype=['Any gibberish works here as both tf and ivy-tf throw the same error'],
    )
