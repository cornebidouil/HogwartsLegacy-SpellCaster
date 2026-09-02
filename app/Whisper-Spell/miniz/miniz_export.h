#pragma once

/* miniz_export.h - Export definitions for miniz library
 * For static linking, MINIZ_EXPORT is defined as empty.
 * For DLL builds, this would contain __declspec(dllexport/dllimport).
 */

#ifndef MINIZ_EXPORT
#  ifdef MINIZ_STATIC_DEFINE
#    define MINIZ_EXPORT
#  else
#    define MINIZ_EXPORT
#  endif
#endif
