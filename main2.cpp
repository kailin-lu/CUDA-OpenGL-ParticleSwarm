#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <sstream>
#include <random>
#include <stdlib.h>
#include "external/glm/glm.hpp"
#include "external/glm/gtc/matrix_transform.hpp"
#include "external/glm/gtc/type_ptr.hpp"

#include "kernel.h"
#include <cuda_runtime.h> 
#include <cuda_gl_interop.h>

#include "shader.h"
#include "camera.h"

/***********************
 * Constants and setup *
 ***********************/ 
const int SCR_HEIGHT = 600; 
const int SCR_WIDTH = 800; 
const int surfaceN = 50; 
const int N = surfaceN * surfaceN; 
float step = 2.0f / static_cast <float>(surfaceN); 
std::string projectName = "OpenGL Project"; 
GLFWwindow *window; 

// Buffers for drawing function surface 
GLuint VAO, VBO, EBO; 
float position[N*3]; 
unsigned int position_indices[N * 4]; 

// Buffers for CUDA particle data 
cudaDeviceProp prop; 
GLuint VAOparticles, VBOparticles; 
cudaGraphicsResource *VBOparticles_CUDA; 
const int NParticles = 2000; 



// camera
Camera camera(glm::vec3(0.0f, 0.0f, 3.0f)); 
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f;


/**************
 * Prototypes *
 **************/ 
void initWindow(std::string projectName);
void mainLoop(Shader shaderSurface, Shader shaderPoints); 
float func(float x, float y);
void initSurface(); 
void initLines(); 
void initCUDA(); 
void generateIndices(); 
void end(); 

void processInput(GLFWwindow *window); 
void framebuffer_size_callback(GLFWwindow* window, int width, int height); 
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);


/*****************
 * Main function *
 *****************/ 
int main(int argc, char *argv[]) {
    // Create a new window context pointer 
    initWindow(projectName); 

    // Check if window was created 
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl; 
        glfwTerminate(); 
        return -1; 
    }
    glfwMakeContextCurrent(window); 
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback); 
    
    // Check if glad loads all OpenGL function pointers 
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl; 
        return -1;
    }

    // Register CUDA resources 
    initCUDA(); 

    // Initialize function positions 
    initSurface(); 

    Shader shaderSurface("shaders/vertex.glsl", "shaders/fragment.glsl"); 
    Shader shaderPoints("shaders/vertex.glsl", "shaders/fragment_particle.glsl"); 
    mainLoop(shaderSurface, shaderPoints);

    end(); 
    return 0; 
}


void initWindow(std::string projectName) {
    glfwInit(); 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,  3); 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); 
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, projectName.c_str(), NULL, NULL); 
}


void initCUDA() {
    cudaSetDevice(0);
    cudaGetDeviceProperties(&prop, 0); 
    getCUDAError("No device properties"); 
    
    // Initialize randomly 
    std::default_random_engine generator; 
    std::uniform_real_distribution<float> distribution(-1.0,1.0); 

    float vertices[3 * NParticles];  
    for (int i = 0; i < NParticles; i++) {
        vertices[3 * i + 0] = distribution(generator);
        vertices[3 * i + 1] = distribution(generator);
        vertices[3 * i + 2] = func(vertices[3 * i + 0], vertices[3 * i + 1]);
    }

    glGenVertexArrays(1, &VAOparticles); 
    glGenBuffers(1, &VBOparticles); 
    
    glBindVertexArray(VAOparticles); 
    glBindBuffer(GL_ARRAY_BUFFER, VBOparticles); 
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW); 
    
    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); 
    glEnableVertexAttribArray(0);

    // cudaGraphicsGLRegisterBuffer(&VBOparticles_CUDA, VBOparticles, cudaGraphicsMapFlagsWriteDiscard);
    // getCUDAError("CUDA Graphics GL Register Buffer"); 
}


void mainLoop(Shader shaderSurface, Shader shaderPoints) {
    // for calculating framerate 
    int frame = 0; 
    double fps = 0.; 
    float lasttime = 0.0;
    float time = 0.0; 
    float lastFrame = 0.0; 
    
    // main loop 
    while (!glfwWindowShouldClose(window)) {
        // process input 
        processInput(window);

        frame++; 
        time = glfwGetTime();
        deltaTime = time - lastFrame; 
        lastFrame = time; 

        if (time - lasttime > 1.0) {
            // calculate fps  
            fps = frame / (time-lasttime); 
            lasttime = time; 
            frame = 0; 
        }

        std::ostringstream ss;
        ss << projectName << " "; 
        ss << " " << prop.name << " ";
        ss.precision(1); 
        ss << std::fixed << fps << " fps"; 
        glfwSetWindowTitle(window, ss.str().c_str());
        
        // Render 
        glClearColor(0.0f, 0.1f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

        shaderSurface.use(); 

        // Set camera 
        glm::mat4 model = glm::mat4(1.0f); 
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix(); 

        shaderSurface.setMat4("model", model); 
        shaderSurface.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        shaderSurface.setMat4("view", view);
        
        // draw surface 
        glBindVertexArray(VAO); 
        glLineWidth(0.1f); 
        glDrawElements(GL_LINE_STRIP_ADJACENCY, 3*N, GL_UNSIGNED_INT, 0); 

        // Set particle shader 
        shaderPoints.use(); 
        shaderPoints.setMat4("model", model); 
        shaderPoints.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        shaderPoints.setMat4("view", view);

        //  draw particles 
        glBindVertexArray(VAOparticles); 
        glPointSize(2.0f); 
        glDrawArrays(GL_POINTS, 0, NParticles); 

        glfwSwapBuffers(window); 
        glfwPollEvents(); 
    }
}

// Initialize function surface 
void initSurface() {
    initLines(); 
    generateIndices(); 
    glGenVertexArrays(1, &VAO); 
    glGenBuffers(1, &VBO); 
    glGenBuffers(1, &EBO); 
    
    glBindVertexArray(VAO); 
    glBindBuffer(GL_ARRAY_BUFFER, VBO); 
    glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO); 
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(position_indices), position_indices, GL_STATIC_DRAW); 
 
    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); 
    glEnableVertexAttribArray(0);
    
    glEnable(GL_DEPTH_TEST); 
    glEnable(GL_PRIMITIVE_RESTART); 
    glad_glPrimitiveRestartIndex(0xffff); 
}

// Initialize values in the function lines 
void initLines() {
    float i_pos = -1.0f;
    float j_pos = -1.0f; 

    for (int i = 0; i < surfaceN; i++) {
        for (int j = 0; j < surfaceN; j++) {
            position[3 * (surfaceN * i + j) + 0] = i_pos;  // x
            position[3 * (surfaceN * i + j) + 1] = j_pos;  // y
            position[3 * (surfaceN * i + j) + 2] = func(i_pos, j_pos);  // z
            j_pos += step; 
        }
        j_pos = -1.0f;
        i_pos += step; 
    }
}

// Draw lines across x axis and y axis to create surface 
void generateIndices() {
    int k = 0; 
    for (int i = 0; i < surfaceN; i++) {
        for (int j = 0; j < surfaceN; j++) {
            position_indices[k] = surfaceN * j + i; 
            k += 1; 
        }
        position_indices[k] = 0xffff; 
        k += 1; 
    }

    for (int i = 0; i < surfaceN; i++) {
        for (int j = 0; j < surfaceN; j++) {
            position_indices[k] = surfaceN * i + j; 
            k += 1;
        }
        position_indices[k] = 0xffff; 
        k += 1; 
    }

}

// // return index of position array where first two values match x and y
// int find(float x, float y) {
//     for (int i = 0; i < N; i++) {
//         if (position[3 * i + 0] == x && position[3 * i + 1] == y) {
//             return i; 
//         }
//     }
//     return -1; 
// }

// parabaloid 
float func(float x, float y) {
    return pow(x, 2) + pow(y, 2); 
}

// // rastrigin 
// float func(float x, float y) {
//     int A = 10; 
//     int n = 2; 
//     return A * 2 + (pow(x, 2) - A * cos(2 * M_PI * x)) + (pow(y, 2) - A * cos(2 * M_PI * y));
// }

// callbacks
void processInput(GLFWwindow *window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true); 
    
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}


void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}


void end() {
    cudaGraphicsUnregisterResource(VBOparticles_CUDA);
    glDeleteVertexArrays(1, &VAO);
    glDeleteVertexArrays(1, &VAOparticles);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &VBOparticles); 
    glfwTerminate(); 
}