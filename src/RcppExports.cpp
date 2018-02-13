// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// get_nearest_neighbors
void get_nearest_neighbors(arma::mat X, arma::mat& X_dist, arma::imat& X_inds, int k);
RcppExport SEXP _rmi_get_nearest_neighbors(SEXP XSEXP, SEXP X_distSEXP, SEXP X_indsSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type X_dist(X_distSEXP);
    Rcpp::traits::input_parameter< arma::imat& >::type X_inds(X_indsSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    get_nearest_neighbors(X, X_dist, X_inds, k);
    return R_NilValue;
END_RCPP
}
// nearest_neighbors
Rcpp::List nearest_neighbors(arma::mat data, int k);
RcppExport SEXP _rmi_nearest_neighbors(SEXP dataSEXP, SEXP kSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    rcpp_result_gen = Rcpp::wrap(nearest_neighbors(data, k));
    return rcpp_result_gen;
END_RCPP
}
// knn_mi
double knn_mi(arma::mat data, Rcpp::NumericVector splits, const Rcpp::List& options);
RcppExport SEXP _rmi_knn_mi(SEXP dataSEXP, SEXP splitsSEXP, SEXP optionsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type data(dataSEXP);
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type splits(splitsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::List& >::type options(optionsSEXP);
    rcpp_result_gen = Rcpp::wrap(knn_mi(data, splits, options));
    return rcpp_result_gen;
END_RCPP
}
// parse_split_vector
void parse_split_vector(Rcpp::NumericVector splits, arma::icolvec& d, arma::icolvec& start_d, arma::icolvec& end_d);
RcppExport SEXP _rmi_parse_split_vector(SEXP splitsSEXP, SEXP dSEXP, SEXP start_dSEXP, SEXP end_dSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::NumericVector >::type splits(splitsSEXP);
    Rcpp::traits::input_parameter< arma::icolvec& >::type d(dSEXP);
    Rcpp::traits::input_parameter< arma::icolvec& >::type start_d(start_dSEXP);
    Rcpp::traits::input_parameter< arma::icolvec& >::type end_d(end_dSEXP);
    parse_split_vector(splits, d, start_d, end_d);
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_rmi_get_nearest_neighbors", (DL_FUNC) &_rmi_get_nearest_neighbors, 4},
    {"_rmi_nearest_neighbors", (DL_FUNC) &_rmi_nearest_neighbors, 2},
    {"_rmi_knn_mi", (DL_FUNC) &_rmi_knn_mi, 3},
    {"_rmi_parse_split_vector", (DL_FUNC) &_rmi_parse_split_vector, 4},
    {NULL, NULL, 0}
};

RcppExport void R_init_rmi(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
