/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <sstream>
#include <string>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  // TODO: Set the number of particles. Initialize all particles to first
  // position (based on estimates of
  //   x, y, theta and their uncertainties from GPS) and all weights to 1.
  // Add random Gaussian noise to each particle.
  // NOTE: Consult particle_filter.h for more information about this method (and
  // others in this file).

  if (!is_initialized) {
    // Set the number of particles.
    num_particles = 256;

    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    // Random Gaussian noise distributions
    std::default_random_engine generator;
    std::normal_distribution<double> distribution_x(x, std_x);
    std::normal_distribution<double> distribution_y(y, std_y);
    std::normal_distribution<double> distribution_theta(theta, std_theta);

    for (int i = 0; i < num_particles; i++) {
      // Initialize all particles to first position (based on estimates of
      // x, y, theta and their uncertainties from GPS) and all weights to 1.
      // Add random Gaussian noise to each particle.
      Particle p{i, distribution_x(generator), distribution_y(generator),
                 distribution_theta(generator), 1.0};
      particles.push_back(p);
    }

    is_initialized = true;
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  // TODO: Add measurements to each particle and add random Gaussian noise.
  // NOTE: When adding noise you may find std::normal_distribution and
  // std::default_random_engine useful.
  //  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
  //  http://www.cplusplus.com/reference/random/default_random_engine/

  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  // Random Gaussian noise distributions
  std::default_random_engine generator;
  std::normal_distribution<double> distribution_x(0, std_x);
  std::normal_distribution<double> distribution_y(0, std_y);
  std::normal_distribution<double> distribution_theta(0, std_theta);

  for (Particle& particle : particles) {
    // Calculate measurements and add to each particle
    if (yaw_rate == 0) {
      particle.x += velocity * delta_t * cos(particle.theta);
      particle.y += velocity * delta_t * sin(particle.theta);
    } else {
      particle.x +=
          velocity / yaw_rate *
          (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
      particle.y +=
          velocity / yaw_rate *
          (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
      particle.theta += yaw_rate * delta_t;
    }

    // Add random Gaussian noise to each particle.
    particle.x += distribution_x(generator);
    particle.y += distribution_y(generator);
    particle.theta += distribution_theta(generator);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
                                     std::vector<LandmarkObs>& observations) {
  // TODO: Find the predicted measurement that is closest to each observed
  // measurement and assign the
  //   observed measurement to this particular landmark.
  // NOTE: this method will NOT be called by the grading code. But you will
  // probably find it useful to
  //   implement this method and use it as a helper during the updateWeights
  //   phase.

  // For each observed measurement,
  for (LandmarkObs& observation : observations) {
    // Find the predicted measurement closest to it
    double dist_min = std::numeric_limits<double>::max();
    int id_min;
    for (const LandmarkObs& prediction : predicted) {
      // Distance between observed and predicted measurements
      double result =
          dist(observation.x, observation.y, prediction.x, prediction.y);

      //  Keep track of the minimum distance and the associated prediction id.
      if (result < dist_min) {
        dist_min = result;
        id_min = prediction.id;
      }
    }

    // Assign observation id to the id associated with the minimum distance
    observation.id = id_min;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const std::vector<LandmarkObs>& observations,
                                   const Map& map_landmarks) {
  // TODO: Update the weights of each particle using a mult-variate Gaussian
  // distribution. You can read
  //   more about this distribution here:
  //   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
  // NOTE: The observations are given in the VEHICLE'S coordinate system. Your
  // particles are located
  //   according to the MAP'S coordinate system. You will need to transform
  //   between the two systems. Keep in mind that this transformation requires
  //   both rotation AND translation (but no scaling). The following is a good
  //   resource for the theory:
  //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
  //   and the following is a good resource for the actual equation to implement
  //   (look at equation 3.33 http://planning.cs.uiuc.edu/node99.html

  // For each particle,
  for (Particle& particle : particles) {
    double x_ptcl = particle.x;
    double y_ptcl = particle.y;
    double theta_ptcl = particle.theta;

    // Filter out landmarks outside the sensor range of the particle.
    vector<LandmarkObs> filtered;
    for (const Map::single_landmark_s landmark : map_landmarks.landmark_list) {
      if (dist(landmark.x_f, landmark.y_f, x_ptcl, y_ptcl) <= sensor_range) {
        filtered.push_back(
            LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // Transform observations from vehicle coordinates to map coordinates.
    vector<LandmarkObs> transformed;
    for (const LandmarkObs& landmark : observations) {
      double t_x =
          cos(theta_ptcl) * landmark.x - sin(theta_ptcl) * landmark.y + x_ptcl;
      double t_y =
          sin(theta_ptcl) * landmark.x + cos(theta_ptcl) * landmark.y + y_ptcl;
      transformed.push_back(LandmarkObs{landmark.id, t_x, t_y});
    }

    // Associate each transformed observation with the prediction closest to it.
    dataAssociation(filtered, transformed);

    // Reinitialize particle weight.
    particle.weight = 1.0;

    for (const LandmarkObs& landmark : transformed) {
      int associated_id = landmark.id;
      double x_obs = landmark.x;
      double y_obs = landmark.y;

      auto associated_landmark = std::find_if(
          filtered.begin(), filtered.end(),
          [=](const LandmarkObs& p) { return p.id == associated_id; });
      double mu_x = associated_landmark->x;
      double mu_y = associated_landmark->y;

      // Calculate normalization term.
      double sig_x = std_landmark[0];
      double sig_y = std_landmark[1];
      double gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

      // Calculate exponent.
      double exponent = (pow(x_obs - mu_x, 2) / pow(sig_x, 2) +
                         pow(y_obs - mu_y, 2) / pow(sig_y, 2)) /
                        2;

      // Calculate weight using normalization terms and exponent.
      double weight = gauss_norm * exp(-exponent);
      // Update particle weight.
      particle.weight *= weight;
    }
  }
}

void ParticleFilter::resample() {
  // TODO: Resample particles with replacement with probability proportional to
  // their weight. NOTE: You may find std::discrete_distribution helpful here.
  //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Get the current weights
  weights.erase(weights.begin(), weights.end());
  for (const Particle& particle : particles) {
    weights.push_back(particle.weight);
  }
  // Get max weight
  auto max_weight = std::max_element(weights.begin(), weights.end());

  std::default_random_engine generator;
  std::uniform_int_distribution<> uniform_int_dist(0, num_particles - 1);
  std::uniform_real_distribution<> uniform_real_dist(0.0, *max_weight);

  // Random index for resampling
  int index = uniform_int_dist(generator);
  // Resample particles with replacement with probability proportional to their
  // weight.
  vector<Particle> resampled_particles;
  double beta = 0.0;
  for (int i = 0; i < num_particles; ++i) {
    beta += uniform_real_dist(generator) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle,
                                         const std::vector<int>& associations,
                                         const std::vector<double>& sense_x,
                                         const std::vector<double>& sense_y) {
  // particle: the particle to assign each listed association, and association's
  // (x,y) world coordinates mapping to
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates

  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;

  return particle;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseX(Particle best) {
  vector<double> v = best.sense_x;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
string ParticleFilter::getSenseY(Particle best) {
  vector<double> v = best.sense_y;
  stringstream ss;
  copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length() - 1);  // get rid of the trailing space
  return s;
}
