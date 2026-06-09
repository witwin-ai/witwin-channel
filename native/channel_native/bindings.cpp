#include <Python.h>

extern PyObject* cn_build_info(PyObject* self, PyObject* args);

static PyMethodDef ChannelNativeMethods[] = {
    {"build_info", cn_build_info, METH_NOARGS, "Return Channel Native build metadata."},
    {nullptr, nullptr, 0, nullptr},
};

static struct PyModuleDef ChannelNativeModule = {
    PyModuleDef_HEAD_INIT,
    "_channel_native",
    "Channel Native C++ extension.",
    -1,
    ChannelNativeMethods,
};

PyMODINIT_FUNC PyInit__channel_native() {
    return PyModule_Create(&ChannelNativeModule);
}
