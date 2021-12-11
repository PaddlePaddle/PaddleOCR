#pragma once

// The way how the function is called
#if !defined(OCR_CALL)
#if defined(_WIN32)
#define OCR_CALL __stdcall
#else
#define OCR_CALL
#endif /* _WIN32 */
#endif /* OCR_CALL */

// The function exported symbols
#if defined _WIN32 || defined __CYGWIN__
#define OCR_IMPORT __declspec(dllimport)
#define OCR_EXPORT __declspec(dllexport)
#define OCR_LOCAL
#else
#if __GNUC__ >= 4
#define OCR_IMPORT __attribute__ ((visibility ("default")))
#define OCR_EXPORT __attribute__ ((visibility ("default")))
#define OCR_LOCAL  __attribute__ ((visibility ("hidden")))
#else
#define OCR_IMPORT
#define OCR_EXPORT
#define OCR_LOCAL
#endif
#endif

#ifdef OCR_EXPORTS // defined if we are building the DLL (instead of using it)
#define OCRAPI_PORT OCR_EXPORT
#else
#define OCRAPI_PORT OCR_IMPORT
#endif // OCR_EXPORTS

#define OCRAPI OCRAPI_PORT OCR_CALL

#define OCRLOCAL OCR_LOCAL OCR_CALL