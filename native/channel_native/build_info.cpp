#include <Python.h>

PyObject* cn_build_info(PyObject* self, PyObject* args) {
    PyObject* info = PyDict_New();
    if (info == nullptr) {
        return nullptr;
    }

    PyObject* backend = PyUnicode_FromString("channel-native");
    if (backend == nullptr) {
        Py_DECREF(info);
        return nullptr;
    }

    const int status =
        PyDict_SetItemString(info, "backend", backend) ||
        PyDict_SetItemString(info, "uses_dr_jit", Py_False) ||
        PyDict_SetItemString(info, "uses_raydn_native", Py_False) ||
        PyDict_SetItemString(info, "cuda_available", Py_False) ||
        PyDict_SetItemString(info, "optix_available", Py_False);
    Py_DECREF(backend);

    if (status) {
        Py_DECREF(info);
        return nullptr;
    }

    return info;
}
