#ifndef DEBUG_H
#define DEBUG_H
 
#include "gl_core_4_5.h"
 
namespace dbg {
/*
 * Register the debug callback using the context capabilities to select between 4.3+ core debug
 * and ARB debug
 */
void register_debug_callback();
 
/*
 * Callbacks for core debug output and debug_output_arb to be selected
 * depending on the context capabilities
 */
void APIENTRY debug_callback(GLenum src, GLenum type, GLuint id, GLenum severity,
	GLsizei len, const GLchar *msg, const GLvoid *user);
 
/*
 * Debug logging function called by our registered callbacks, simply
 * logs the debug messages to stdout
 */
void log_debug_msg(GLenum src, GLenum type, GLuint id, GLenum severity, GLsizei len, const GLchar *msg);
}
 
#endif

