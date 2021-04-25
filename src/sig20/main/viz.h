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
    int m_current_building;
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_trajectories;
    std::vector<int> m_is_reconstruction_status;
    std::vector<std::pair<Eigen::Vector3f, Eigen::Vector3f>> m_trajectories_spline;
    std::vector<std::pair<Eigen::Vector2f, cv::Vec3b>> m_uncertainty_map;
    float m_uncertainty_map_distance;
    Eigen::Vector3f m_pos;
    Eigen::Vector3f m_direction;
    Point_set m_points;
    std::vector<Eigen::Vector4f> m_points_color;
    //CGAL::General_polygon_2<K> m_polygon;
	
	Visualizer()
	{
        m_thread = new std::thread(&Visualizer::run, this);
        //render_loop.join();
        return;
	}
    
    void draw_point_cloud(const Point_set& v_points) {
        glBegin(GL_POINTS);
        glColor3f(1,1,1);
        for (size_t i = 0; i < v_points.size(); ++i) {
            glPointSize(5);

            glVertex3d(v_points.point(i).x(), v_points.point(i).y(), v_points.point(i).z());
        }
        glEnd(); 
        glPointSize(1);
    }
	
    void draw_line(const Eigen::Vector3f& v_min, const Eigen::Vector3f& v_max, int sickness=1,const Eigen::Vector4f& v_color = Eigen::Vector4f(1.f, 0.f, 0.f, 1.f))
	{
        glLineWidth(sickness);
        glColor3f(v_color.x(), v_color.y(), v_color.z());
        pangolin::glDrawLine(v_min.x(), v_min.y(), v_min.z(), v_max.x(), v_max.y(), v_max.z());
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

        draw_line(Eigen::Vector3f(l[0], l[1], l[2]), Eigen::Vector3f(l[0],l[1],h[2]),2,Eigen::Vector4f(0,0,0,1));
        draw_line(Eigen::Vector3f(l[0], l[1], l[2]), Eigen::Vector3f(l[0],h[1],l[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(l[0], l[1], l[2]), Eigen::Vector3f(h[0],l[1],l[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(l[0], h[1], l[2]), Eigen::Vector3f(h[0],h[1],l[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(l[0], h[1], l[2]), Eigen::Vector3f(l[0],h[1],h[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(l[0], h[1], h[2]), Eigen::Vector3f(h[0],h[1],h[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(l[0], h[1], h[2]), Eigen::Vector3f(l[0],l[1],h[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(h[0], l[1], l[2]), Eigen::Vector3f(h[0], h[1], l[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(h[0], l[1], l[2]), Eigen::Vector3f(h[0], l[1], h[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(h[0], h[1], h[2]), Eigen::Vector3f(h[0], l[1], h[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(h[0], h[1], h[2]), Eigen::Vector3f(h[0], h[1], l[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
        draw_line(Eigen::Vector3f(l[0], l[1], h[2]), Eigen::Vector3f(h[0], l[1], h[2]), 2, Eigen::Vector4f(0, 0, 0, 1));
    }

    void run() {
        //pangolin::CreateWindowAndBind("Main", 640, 960);
        pangolin::CreateWindowAndBind("Main", 1600, 960);
        //pangolin::CreateWindowAndBind("Main", 1280, 960);
        glEnable(GL_DEPTH_TEST);
        pangolin::OpenGlRenderState s_cam1(
            pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 99999),
            pangolin::ModelViewLookAt(40, 40, 40, 0, 0, 0, pangolin::AxisZ)
        );
		pangolin::OpenGlRenderState s_cam2(
            pangolin::ProjectionMatrix(1280, 960, 50, 50, 640, 480, 0.2, 99999),
            //pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.2, 99999),
            // pangolin::ProjectionMatrixOrthographic(-1, 1, -1,1, 0.2, 99999),
            pangolin::ModelViewLookAt(-250, 1.f, 40.f, -250, 0, 0, pangolin::AxisZ)
        );
		

        // Create Interactive View in window
        MyHandler handler1(s_cam1, pangolin::AxisZ);
        MyHandler handler2(s_cam2,pangolin::AxisZ);
        pangolin::View& d_cam1 = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0f, -640 / 480.f)
            .SetHandler(&handler1);
        pangolin::View& d_cam2 = pangolin::CreateDisplay()
            .SetBounds(0, 1.0f, 1.0f, 1.f, -640 / 480.f)
			.SetHandler(&handler2);

        pangolin::Display("multi")
            .SetBounds(0.0, 1.0, 0.0, 1.0)
            .SetLayout(pangolin::LayoutEqualHorizontal)
            .AddDisplay(d_cam1)
            .AddDisplay(d_cam2)
		;

		
        while (!pangolin::ShouldQuit()) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glClearColor(0.56f, 0.56f, 0.56f, 1);

            lock();
            d_cam2.Activate(s_cam2);
            for (const auto& item_tile : m_uncertainty_map) {
                const Eigen::Vector2f& position = item_tile.first;
                const cv::Vec3b& item_color = item_tile.second;
                Eigen::Vector4f color;
                color = Eigen::Vector4f((float)item_color[2] / 255.f, (float)item_color[1] / 255.f, (float)item_color[0] / 255.f, 1);

                draw_cube(Eigen::AlignedBox3f(Eigen::Vector3f(position.x() - m_uncertainty_map_distance / 2, position.y() - m_uncertainty_map_distance / 2, -1),
                    Eigen::Vector3f(position.x() + m_uncertainty_map_distance / 2, position.y() + m_uncertainty_map_distance / 2, 1)), color);
            }
        	
            //pangolin::glDrawAxis(1000);
            //draw_point_cloud(m_points);
        	
            d_cam1.Activate(s_cam1);
            //pangolin::glDrawAxis(1000);
            // Render
            //draw_point_cloud(m_points);

        	//Building
        	for (const auto& item_building : m_buildings)
            {
                int index = &item_building - &m_buildings[0];
                if(m_current_building == index)
	                draw_cube(item_building.bounding_box_3d,
	                    Eigen::Vector4f(1.f, 0.f, 0.f, 1.f));
                else if(item_building.passed_trajectory.size()!=0)
                    draw_cube(item_building.bounding_box_3d,
                        Eigen::Vector4f(1.f, 1.f, 1.f, 1.f));
                else
                    draw_cube(item_building.bounding_box_3d,
                        Eigen::Vector4f(0.5f, .5f, .5f, .5f));
            }

        	//View points
            for (const auto& item_trajectory : m_trajectories) {
                draw_cube(Eigen::AlignedBox3f(item_trajectory.first - Eigen::Vector3f(1.f, 1.f, 1.f), item_trajectory.first + Eigen::Vector3f(1.f, 1.f, 1.f)),
                    Eigen::Vector4f(0.f, 1.f, 0.f, 1.f));
                Eigen::Vector3f look_at = item_trajectory.first + item_trajectory.second * 10;
                draw_line(item_trajectory.first, look_at,2,Eigen::Vector4f(0,1,0,1));
            }
            m_trajectories_spline = m_trajectories;
        	//View spline
            for (const auto& item_trajectory: m_trajectories_spline) {
                int index = &item_trajectory - &m_trajectories_spline[0];

                Eigen::Vector4f color(250./255, 157./255, 0./255, 1);
                if (m_is_reconstruction_status[index] == 1)
                    color = Eigen::Vector4f( 23./255, 73./255, 179./255, 1);
                
                //draw_cube(Eigen::AlignedBox3f(item_trajectory.first - Eigen::Vector3f(1.f, 1.f, 1.f), item_trajectory.first + Eigen::Vector3f(1.f, 1.f, 1.f)),
                //    Eigen::Vector4f(0.f, 1.f, 0.f, 1.f));
                glColor3f(color.x(), color.y(), color.z());
            	if(index >=1)
                    pangolin::glDrawLine(item_trajectory.first[0], item_trajectory.first[1], item_trajectory.first[2], 
                        m_trajectories_spline[index -1].first[0], m_trajectories_spline[index - 1].first[1], m_trajectories_spline[index - 1].first[2]);

                glColor3f(0,0,0);
            }
        	// View points
            //for (const auto& id_point : m_points)
            //{
                //Eigen::Vector4f color(1.f, 1.f, 1.f, 1.f);
                //if (m_points_color.size() > 0)
                //    color = m_points_color[&id_point - &*m_points.begin()];
                //Point_3& p = m_points.point(id_point);
                //float radius = 0.1f;
                //draw_cube(Eigen::AlignedBox3f(Eigen::Vector3f(p.x()- radius, p.y()- radius, p.z()- radius), 
            //        Eigen::Vector3f(p.x() + radius, p.y() + radius, p.z() + radius)), color);
            //}
        	// Current Position and orientation
            draw_cube(Eigen::AlignedBox3f(m_pos - Eigen::Vector3f(4.f, 4.f, 4.f), m_pos + Eigen::Vector3f(4.f, 4.f, 4.f)),
                Eigen::Vector4f(1.f, 0.f, 0.f, 1.f));
            Eigen::Vector3f look_at = m_pos + m_direction * 20;
            draw_line(m_pos, look_at, 2, Eigen::Vector4f(0, 1, 0, 1));
        	
            // Uncertainty
            for (const auto& item_tile : m_uncertainty_map) {
                const Eigen::Vector2f& position= item_tile.first;
                const cv::Vec3b& item_color = item_tile.second;
                Eigen::Vector4f color;
                color = Eigen::Vector4f((float)item_color[2]/255.f, (float)item_color[1] / 255.f, (float)item_color[0] / 255.f, 1);

                draw_cube(Eigen::AlignedBox3f(Eigen::Vector3f(position.x() - m_uncertainty_map_distance/2, position.y() - m_uncertainty_map_distance / 2, -1),
                    Eigen::Vector3f(position.x() + m_uncertainty_map_distance / 2, position.y() + m_uncertainty_map_distance / 2, 1)), color);
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