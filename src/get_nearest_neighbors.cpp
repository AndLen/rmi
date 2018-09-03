#include <armadillo>
#include "flann/flann.hpp"

using namespace std;

void get_nearest_neighbors(arma::mat X,
                           arma::mat&  X_dist,
                           arma::imat& X_inds,
                           int k) {
  int d = X.n_cols;
  int N = X.n_rows;
  int K = k + 1;

  arma::mat query(d, N);
  arma::mat temp_q(X.begin(), N, d, false);
  query = arma::trans(temp_q); //transpose
  flann::Matrix<double> q_flann(query.memptr(), N, d);
  flann::Matrix<double> ref_flann(query.memptr(), N, d);

  //Setting the flann index params
  flann::IndexParams params;
  params = flann::LinearIndexParams();

  flann::Index<flann::MaxDistance<double> > index(ref_flann, params);
  index.buildIndex();
  flann::Matrix<int> indices_flann(new int[N * K], N, K);
  flann::Matrix<double> dists_flann(new double[N * K], N, K);
  flann::SearchParams search_params;

  //search_params.cores = 10;
  //search_params.checks = 100;

  index.knnSearch(q_flann, indices_flann, dists_flann, K, search_params);

  arma::imat indices(indices_flann.ptr(), K, N, true);
  arma::mat dists(dists_flann.ptr(), K, N, true);

  delete[] indices_flann.ptr();
  delete[] dists_flann.ptr();

  X_inds = indices.t();
  X_dist = dists.t();

  return;
}