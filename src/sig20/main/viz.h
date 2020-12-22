#pragma once
#include <pangolin/pangolin.h>
#include <thread>
#include <mutex>

#include "model_tools.h"
#include "building.h"
#include "cgal_tools.h"

struct PANGOLIN_EXPORT MyHandler : pangolin::Handler3D {
    MyHandler(pangolin::OpenGlRenderState& cam_state, pangolin::AxisDirection enforce_up = pangolin::AxisNone, float trans_scale = 0.01f, float zoom_fraction = PANGO_DFLT_HANDLER3D_ZF) :
        Handler3D(cam_state, enforce_up, trans_scale, zoom_fraction){}
    //void Mouse(pangolin::View&, pangolin::MouseButton button, int x, int y, bool pressed, int button_state)
    //{
	    
    //}
    //void MouseMotion(pangolin::View&, int x, int y, int button_state);

};

class Visualizer
{
public:
    std::thread* m_thread;
    std::mutex m_mutex;
    std::vector<Building> m_buildings;
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_trajectories;
    Eigen::Vector3f m_pos;
    Eigen::Vector3f m_direction;
    Point_set m_points;
    std::vector<Eigen::Vector4f> m_points_color;
	
	Visualizer()
	{
        m_thread = new std::thread(&Visualizer::run, this);
        //render_loop.join();

        return;
	}

	void draw_cube(const Eigen::AlignedBox<float, 3>& box,const Eigen::Vector4f& v_color=Eigen::Vector4f(1.f,0.f,0.f,1.f))
	{
        const Eigen::Matrix<float, 3, 1> l = box.min().template cast<float>();
        const Eigen::Matrix<float, 3, 1> h = box.max().template cast<float>();
        const GLfloat verts[] = {
            l[0],l[1],h[2],  h[0],l[1],h[2],  l[0],h[1],h[2],  h[0],h[1],h[2],  // FRONT
            l[0],l[1],l[2],  l[0],h[1],l[2],  h[0],l[1],l[2],  h[0],h[1],l[2],  // BACK
            l[0],l[1],h[2],  l[0],h[1],h[2],  l[0],l[1],l[2],  l[0],h[1],l[2],  // LEFT
            h[0],l[1],l[2],  h[0],h[1],l[2],  h[0],l[1],h[2],  h[0],h[1],h[2],  // RIGHT
            l[0],h[1],h[2],  h[0],h[1],h[2],  l[0],h[1],l[2],  h[0],h[1],l[2],  // TOP
            l[0],l[1],h[2],  l[0],l[1],l[2],  h[0],l[1],h[2],  h[0],l[1],l[2]   // BOTTOM
        };

        glVertexPointer(3, GL_FLOAT, 0, verts);
        glEnableClientState(GL_VERTEX_ARRAY);

        glColor4f(v_color[0], v_color[1], v_color[2], v_color[3]);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        glDrawArrays(GL_TRIANGLE_STRIP, 4, 4);

        glColor4f(v_color[0], v_color[1], v_color[2], v_color[3]);
        glDrawArrays(GL_TRIANGLE_STRIP, 8, 4);
        glDrawArrays(GL_TRIANGLE_STRIP, 12, 4);

        glColor4f(v_color[0], v_color[1], v_color[2], v_color[3]);
        glDrawArrays(GL_TRIANGLE_STRIP, 16, 4);
        glDrawArrays(GL_TRIANGLE_STRIP, 20, 4);

        glDisableClientState(GL_VERTEX_ARRAY);
	}

    void run() {
        // fetch the context and bind it to this thread
        pangolin::CreateWindowAndBind("Main", 1000, 800);

        // we manually need to restore the properties of the context
        glEnable(GL_DEPTH_TEST);

        // Define Projection and initial ModelView matrix
        pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 99999),
            pangolin::ModelViewLookAt(40, 40, 40, 0, 0, 0, pangolin::AxisZ)
        );

        // Create Interactive View in window
        MyHandler handler(s_cam,pangolin::AxisZ);
        pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(&handler);

        while (!pangolin::ShouldQuit()) {
            // Clear screen and activate view to render into
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.75f, 0.75f, 0.75f, 1);
            d_cam.Activate(s_cam);

            pangolin::glDrawAxis(1000);
            // Render
            lock();
            draw_cube(Eigen::AlignedBox3f(m_pos - Eigen::Vector3f(2.f, 2.f, 2.f), m_pos + Eigen::Vector3f(2.f, 2.f, 2.f)),
                Eigen::Vector4f(1.f, 0.f, 0.f, 1.f));
            Eigen::Vector3f look_at = m_pos + m_direction*10;
        	pangolin::glDrawLine(m_pos[0], m_pos[1], m_pos[2], look_at[0], look_at[1], look_at[2]);
            for (const auto& item_building : m_buildings)
            {
                draw_cube(item_building.bounding_box_3d,
                    Eigen::Vector4f(1.f, 1.f, 1.f, 0.75f));
            }
            for (const auto& item_trajectory : m_trajectories) {
                draw_cube(Eigen::AlignedBox3f(item_trajectory.first - Eigen::Vector3f(1.f, 1.f, 1.f), item_trajectory.first + Eigen::Vector3f(1.f, 1.f, 1.f)),
                    Eigen::Vector4f(0.f, 1.f, 0.f, 1.f));
                Eigen::Vector3f look_at = item_trajectory.first + item_trajectory.second * 10;
                pangolin::glDrawLine(item_trajectory.first[0], item_trajectory.first[1], item_trajectory.first[2], look_at[0], look_at[1], look_at[2]);
            }
        	if(m_points.size()>0)
        	{
                for (const auto& id_point : m_points)
                {
                    Eigen::Vector4f color(1.f, 1.f, 1.f, 1.f);
                    if (m_points_color.size() > 0)
                        color = m_points_color[&id_point - &*m_points.begin()];
                    Point_3& p = m_points.point(id_point);
                    float radius = 0.1f;
                    draw_cube(Eigen::AlignedBox3f(Eigen::Vector3f(p.x()- radius, p.y()- radius, p.z()- radius), 
                        Eigen::Vector3f(p.x() + radius, p.y() + radius, p.z() + radius)), color);
                }
        	}
            unlock();

            // Swap frames and Process Events
            pangolin::FinishFrame();
        }

        // unset the current context from the main thread
        pangolin::GetBoundWindow();
    }

	void lock()
	{
        m_mutex.lock();
	}

    void unlock() {
        m_mutex.unlock();
    }

	
};