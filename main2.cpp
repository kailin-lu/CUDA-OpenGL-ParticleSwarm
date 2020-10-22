#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <string>
#include <sstream>
// #include <cuda_runtime.h> 
// #include <cuda_gl_interop.h>
#include <random>
#include "external/glm/glm.hpp"
#include "external/glm/gtc/matrix_transform.hpp"
#include "external/glm/gtc/type_ptr.hpp"

// #include "kernel.h"
#include "shader.h"
#include "camera.h"

/***********************
 * Constants and setup *
 ***********************/ 
const int SCR_HEIGHT = 600; 
const int SCR_WIDTH = 800; 
const int surfaceN = 25; 
const int N = surfaceN * surfaceN; 
std::string projectName = "OpenGL Project"; 
GLFWwindow *window; 
// cudaDeviceProp prop; 
GLuint VAO, VBO; 
float position[N*3]; 

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
void mainLoop(Shader shaderPoints); 
float func(float x, float y);
void initLines(); 

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

    // Initialize function positions 
    initLines(); 
    glGenVertexArrays(1, &VAO); 
    glGenBuffers(1, &VBO); 
    
    glBindVertexArray(VAO); 
    glBindBuffer(GL_ARRAY_BUFFER, VBO); 
    glBufferData(GL_ARRAY_BUFFER, sizeof(position), position, GL_STATIC_DRAW);
    glVertexAttribPointer(0,3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0); 
    glEnableVertexAttribArray(0);
    
    glEnable(GL_DEPTH_TEST); 
    Shader shaderPoints("shaders/vertex.glsl", "shaders/fragment.glsl"); 
    mainLoop(shaderPoints);

    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glfwTerminate(); 
    return 0; 
}


void initWindow(std::string projectName) {
    glfwInit(); 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,  3); 
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); 
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, projectName.c_str(), NULL, NULL); 
}

void mainLoop(Shader shaderPoints) {
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
        // ss << " " << prop.name << " ";
        ss.precision(1); 
        ss << std::fixed << fps << " fps"; 
        glfwSetWindowTitle(window, ss.str().c_str());
        
        // Render 
        glClearColor(0.0f, 0.1f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); 

        glBindVertexArray(VAO); 
        glPointSize(1.0f); 
        // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glDrawArrays(GL_POINTS, 0, N);

        shaderPoints.use(); 

        // Set camera 
        glm::mat4 view          = glm::mat4(1.0f); // make sure to initialize matrix to identity matrix first
        glm::mat4 projection    = glm::mat4(1.0f);
        glm::mat4 model = glm::mat4(1.0f); 
        projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        view = camera.GetViewMatrix(); 

        shaderPoints.setMat4("model", model); 
        shaderPoints.setMat4("projection", projection); // note: currently we set the projection matrix each frame, but since the projection matrix rarely changes it's often best practice to set it outside the main loop only once.
        shaderPoints.setMat4("view", view);
        glfwSwapBuffers(window); 
        glfwPollEvents(); 
    }
}

// Initialize values in the function lines 
void initLines() {
    float step = 2.0f / static_cast <float>(surfaceN); 
    float i_pos = -1.0f;
    float j_pos = -1.0f; 
    // std::default_random_engine generator; 
    // std::uniform_real_distribution<float> distribution(-1.0,1.0); 

    for (int i = 0; i < surfaceN; i++) {
        for (int j = 0; j < surfaceN; j++) {
            // i_pos = distribution(generator); 
            // j_pos = distribution(generator); 
            position[3 * (surfaceN * i + j) + 0] = i_pos;  // x
            position[3 * (surfaceN * i + j) + 1] = j_pos;  // y
            position[3 * (surfaceN * i + j) + 2] = func(i_pos, j_pos);  // z
            j_pos += step; 
        }
        j_pos = -1.0f;
        i_pos += step; 
    }
}

float func(float x, float y) {
    return pow(x, 2) + pow(y, 2); 
}

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
