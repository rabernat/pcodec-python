use pco::standalone::{auto_compress, simple_decompress_into};
use pco::DEFAULT_COMPRESSION_LEVEL;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::{pymodule, FromPyObject, PyModule, PyResult, Python, PyObject};
use pyo3::types::PyBytes;

#[pymodule]
fn pcodec(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    // This is a pure function (no mutations of incoming data).
    // You can see this as the python array in the function arguments is readonly.
    // The object we return will need ot have the same lifetime as the Python.
    // Python will handle the objects deallocation.
    // We are having the Python as input with a lifetime parameter.
    // Basically, none of the data that comes from Python can survive
    // longer than Python itself. Therefore, if Python is dropped, so must our Rust Python-dependent variables.
    #[pyfn(m)]
    fn compress<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<f64>) -> PyObject {
        // Here we have a numpy array of dynamic size. But we could restrict the
        // function to only take arrays of certain size
        // e.g. We could say PyReadonlyArray3 and only take 3 dim arrays.
        // These functions will also do type checking so a
        // numpy array of type np.float32 will not be accepted and will
        // yield an Exception in Python as expected
        let array = x.as_array();
        let slice = array.as_slice().unwrap();
        let compressed: Vec<u8> = auto_compress(&slice, DEFAULT_COMPRESSION_LEVEL);
        PyBytes::new(py, &compressed).into()
    }

    #[pyfn(m)]
    fn decompress<'py>(compressed: &PyBytes, out: &PyArrayDyn<f64>) {
        let src: &[u8] = compressed.extract().unwrap();
        let mut out_rw = out.readwrite();
        let dst = out_rw.as_slice_mut().expect("failed to get mutable slice");
        let _result = simple_decompress_into::<f64>(src, dst).expect("failed to decompress");
    }


    Ok(())
}
