#ifndef XRD_OCL_COMPILER_HPP
#define XRD_OCL_COMPILER_HPP

//
// This include file will deal with compiler differences
//

#if not defined(__cplusplus) || __cplusplus < 201103L
#  error "A C++11 compiler is needed for this code."
#endif



//
// restricted pointers
// restrict keyword will be defined if need to the appropriate version for the
// compiler, or just left undefined if no support for the feature is present.
// Note that restrict does not appear in the C++ standard.
//

#if defined(__GNUC__)
// This applies to both, gcc, an clang
#  define restrict __restrict__
#elif defined(_MSVC_VER)
#  define restrict __restrict
#else
#  define restrict
#endif

#endif // XRD_OCL_COMPILER_HPP
