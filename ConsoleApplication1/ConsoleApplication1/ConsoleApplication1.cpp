// ConsoleApplication1.cpp : This file contains the 'main' function. Program execution begins and ends there.
//


#include "pch.h"
#include <iomanip>
#include <iostream>
#include <cmath>
#include <limits>
#include <Eigen/Dense>
#include <Unsupported/Eigen/MPRealSupport>
#include "Timer.h"

extern "C" {
	int daxpy_(const int* n, const double* da, const double* dx, const int* incx, double* dy, const int* incy);
}

template<typename Scalar>
void eigenTypeDemo(unsigned int dim) {
	using dynMat_t = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

	using dynColVec_t = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
	using dynRowVec_t = Eigen::Matrix<Scalar, 1, Eigen::Dynamic>;

	using index_t = typename dynMat_t::Index;
	using entry_t = typename dynMat_t::Scalar;

	dynColVec_t colvec(dim);
	dynRowVec_t rowvex(dim);

	for (index_t i = 0; i < colvec.size(); ++i) {
		colvec(i) = (Scalar)i;
	}

	for (index_t i = 0; i < rowvex.size(); ++i) {
		rowvex(i) = (Scalar)1 / (i + 1);
	}

	colvec[0] = (Scalar)3.14; rowvex(dim - 1) = (Scalar)2.718;

	dynMat_t vecprod = colvec * rowvex;
	const int nrows = vecprod.rows();
	const int ncols = vecprod.cols();
}

void matArray(int nrows, int ncols) {
	Eigen::MatrixXd m1(nrows, ncols), m2(nrows, ncols);
	for (int i = 0; i < m1.rows();i++) {
		for (int j = 0; j < m1.cols(); j++) {
			m1(i, j) = (double)(i + 1) / (j + 1);
			m2(i, j) = (double)(i + 1) / (j + 1);
		}
	}

	Eigen::MatrixXd m3 = (m1.array() * m2.array()).matrix();
	Eigen::MatrixXd m4(m1.cwiseProduct(m2));

	std::cout << "Log(m1)=" << std::endl << log(m1.array()) << std::endl;
	std::cout <<  (m1.array() > 3).count() << " entries of m1 > 3 " << std::endl;
}

void storageOrder(int nrows = 6, int ncols = 7) {
	std::cout << "Different matrix storage layouts in Eigen " << std::endl;
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> mcm(nrows, ncols);
	Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mrm(nrows, ncols);
	
	for (int l = 1, i = 0; i < nrows; i++) {
		for (int j = 0; j < ncols; j++, l++) {
			mcm(i, j) = mrm(i, j) = l;
		}
	}

	std::cout << "Matrix mrm = " << std::endl << mrm << std::endl;
	std::cout << "mcm linear = ";
	for (int i = 0; i < mcm.size(); i++) std::cout << mcm(i) << "," ;
	std::cout << std::endl;

	std::cout << "mrm linear = ";
	for (int i = 0; i < mrm.size(); i++) std::cout << mrm(i) << ",";
	std::cout << std::endl;
}

void initializeMatrix(int cols=5, int rows=6) {
	Eigen::MatrixXd A(cols, rows);
	Eigen::MatrixXd B = Eigen::MatrixXd::Zero(cols, rows);
	Eigen::MatrixXd C = Eigen::MatrixXd::Ones(cols, rows);
	Eigen::MatrixXd D = Eigen::MatrixXd::Constant(cols, rows, 7.5);
	Eigen::MatrixXd E = Eigen::MatrixXd::Random(cols, rows);
	Eigen::MatrixXd I = Eigen::MatrixXd::Identity(cols, rows);

	std::cout << "size of A = (" << B.rows() << "," << B.cols() << ")" << std::endl;
}

template<typename MatType>
void blockAccess(Eigen::MatrixBase<MatType> &M) {
	using index_t = typename Eigen::MatrixBase<MatType>::Index;
	using entry_t = typename Eigen::MatrixBase<MatType>::Scalar;
	const index_t nrows(M.rows());
	const index_t ncols(M.cols());

	std::cout << "Matrix M=" << std::endl << M << std::endl;
	index_t p = nrows / 2, q = ncols / 2;
	for (index_t i = 0; i < min(p, q);i++) {
		std::cout << "Block (" << i << "," << i << "," << p << "," << q << ") = " << M.block(i, i, p, q) << std::endl;

	}
	M.block(1, 1, p, q) += Eigen::MatrixBase<MatType>::Constant(p, q, 1, 0);
	std::cout << "M = " << std::endl << M << std::endl;

	Eigen::MatrixXd B = M.block(1,1,p,q);
	std::cout << "Isolated modified block = " << std::endl << B << std::endl;
	
	std::cout << p << " top rows of m " << M.topRows(p) << std::endl;
	std::cout << p << " bottom rows of m " << M.bottomRows(p) << std::endl;
	std::cout << q << " left cols of m " << M.leftCols(p) << std::endl;
	std::cout << q << " right cols of m " << M.rightCols(p) << std::endl;

	const Eigen::MatrixXd T = M.template triangularView<Eigen::Upper>();
	std::cout << "Upper triangualar part = " << std::endl << T << std::endl;

	M.template triangularView<Eigen::Lower>() += -1.5;
	std::cout << "Matrix M = " << std::endl << M << std::endl;
}

template<typename MatType>
void reshapeTest(Eigen::MatrixBase<MatType> &M) {
	using index_t = typename Eigen::MatrixBase<MatType>::Index;
	using entry_t = typename Eigen::MatrixBase<MatType>::Scalar;
	const index_t nsize(M.size());

	if ((nsize % 2) == 0) {
		entry_t *Mdat = M.data();
		Eigen::Map<Eigen::Matrix<entry_t, Eigen::Dynamic, Eigen::Dynamic>> R(Mdat, 2, nsize / 2);
		Eigen::Matrix<entry_t, Eigen::Dynamic, Eigen::Dynamic> S =
			Eigen::Map<Eigen::Matrix<entry_t, Eigen::Dynamic, Eigen::Dynamic>>(Mdat, 2, nsize / 2);

		std::cout << "Matrix M = " << std::endl << M << std::endl;
		std::cout << "Reshape to " << R.rows() << 'x' << R.cols() << ' = ' << std::endl << R << std::endl;

		R += -1.5;
		std::cout << "Scaled (!) Matrix M = " << std::endl << M << std::endl;
		std::cout << "Matrix S = " << std::endl << S << std::endl;
	}
 }

void blasbasedsaxpy() {
	const int n = 5;
	const int incx = 1;
	const int incy = 1;
	double alpha = 2.5;

	double* x = new double[n];
	double* y = new double[n];

	for (size_t i = 0; i < n; i++) {
		x[i] = 3.1415 * i;
		y[i] = 1.0 / (double)(i + 1);
	}

	std::cout << "x=["; for (size_t i = 0; i < n; i++) std::cout << x[i] << " ";
	std::cout << "]" << std::endl;

	std::cout << "y=["; for (size_t i = 0; i < n; i++) std::cout << y[i] << " ";
	std::cout << "]" << std::endl;

	//daxpy_(&n, &alpha, x, &incx, y, &incy);

	std::cout << "y = " << alpha << "* x + y = ]";
	for (int i = 0; i < n; i++) std::cout << y[i] << " "; std::cout << "]" << std::endl;
}

void mmeigenmkl() {
	int nruns = 3, minExp = 6, maxExp = 13;
	Eigen::MatrixXd timing(maxExp - minExp + 1, 2);
	for (int p = 0; p <= maxExp - minExp; ++p) {
		Timer t1;
		int n = std::pow(2, minExp + p);
		Eigen::MatrixXd A = Eigen::MatrixXd::Random(n, n);
		Eigen::MatrixXd B = Eigen::MatrixXd::Random(n, n);
		Eigen::MatrixXd C = Eigen::MatrixXd::Zero(n, n);

		for (int i = 0; i < nruns; ++i)
		{
			t1.Start();
			C = A * B;
			t1.Stop();
		}
		timing(p, 0) = n; //timing(p, 1) = t1.min();
	}

	std::cout << std::scientific;
	std::cout << std::setprecision(3);
	std::cout << timing << std::endl;
}

Eigen::MatrixXd dottenstiming() {
	const int nruns = 3, minExp = 2, maxExp = 13;
	
	Eigen::MatrixXd timings(maxExp - minExp + 1, 3);
	for (int i = 0; i < maxExp - minExp; ++i)
	{
		Timer tfool, tsmart;
		const int n = std::pow(2, minExp + i);
		Eigen::VectorXd a = Eigen::VectorXd::LinSpaced(n,1,n);
		Eigen::VectorXd b = Eigen::VectorXd::LinSpaced(n, 1, n).reverse();
		Eigen::VectorXd x = Eigen::VectorXd::Random(n, 1),y(n);

		for (int j = 0; j < nruns; ++j)
		{
			tfool.Start(); y = (a*b.transpose())*x; tfool.Stop();
			tsmart.Start(); y = a * b.dot(x); tsmart.Stop();
		}

		timings(i, 0) = n;
		//timings(i, 1) = tsmart.min();
		//timings(i, 2) = tfool.min();
	}
	return timings;
}

template<class Vec, class Mat> void Irtrimulteff(const Mat& A, const Mat& B, const Vec& x, const Vec& y) {
	const int n = A.rows(), p = A.cols();
	assert(n == B.rows() && p == B.cols());
	for (int i = 0; i < p; ++i)
	{
		Vec tmp = (B.col(i).array() * x.array()).matrix().reverse();
		std::partial_sum(tmp.data(), tmp.data() + n, tmp.data());
		y += (A.col(i).array() * tmp.reverse().array()).matrix();
	}
}

template <class Matrix> Matrix gramschmidt(const  Matrix& A) {
	Matrix Q = A;
	Q.col(0).normalize();
	for (unsigned int j = 1; j < A.cols(); ++j) {
		Q.col(j) -= Q.leftCols(j) * (Q.leftCols(j).transpose() * A.col(j));
		if (Q.col(j).norm() <= 10e-9 * A.col(j).norm()) {
			std::cerr << "Gram-Schmidt failed: A has lin. dep columns." << std::endl;
			break;
		}
		else {
			Q.col(j).normalize();
		}
	}

	return Q;
}

void gsroundoff(Eigen::MatrixXd& A) {
	Eigen::MatrixXd Q = gramschmidt(A);
	std::cout << std::setprecision(4) << std::fixed << "I = " << std::endl << Q.transpose()*Q << std::endl;
	//HouseholderQR<Eigen::MatrixXd> qr(A.rows(), A.cols());
	//qr.compute(A); Eigen::MatrixXd Q1 = qr.householderQ();

	//std::cout << "I1 = " << std::endl << Q1.transpose()*Q1 << std:endl;

	//Eigen::MatrixXd R1 = qr.matrixQR().triangularView<Upper>();
	//std::cout << std::scientific << "A-Q1*R1 = " << std::endl << A - Q1 * R1 << std:endl;

}

void CharacteristicOfDoubleNumbers() {
	std::cout << std::numeric_limits<double>::is_iec559 << std::endl
		<< std::defaultfloat << std::numeric_limits<double>::min() << std::endl
		<< std::hexfloat << std::numeric_limits<double>::min() << std::endl
		<< std::defaultfloat << std::numeric_limits<double>::max() << std::endl
		<< std::hexfloat << std::numeric_limits<double>::max() << std::endl;
}

void RoundoffErrors() {
	std::cout.precision(15);
	double a = 4.0 / 3.0, b = a - 1, c = 3 * b, e = 1 - c;
	std::cout << e << std::endl;
	a = 1012.0 / 113.0; b = a - 9; c = 113 * b; e = 5 + c;
	std::cout << e << std::endl;
	a = 83810206.0 / 6789.0; b= a - 12345; c = 6789 * b; e = c - 1;
	std::cout << e << std::endl;
}

void EPS() {
	std::cout.precision(15);
	std::cout << std::numeric_limits<double>::epsilon() << std::endl;

	std::cout.precision(25);
	double eps = std::numeric_limits<double>::epsilon();

	std::cout << std::fixed << 1.0 + 0.5*eps << std::endl << 1.0 - 0.5*eps << std::endl << (1.0 + 2 / eps) - 2 / eps << std::endl;
}

void overflowunderflow() {
	std::cout.precision(15);
	double min = std::numeric_limits<double>::min();
	double res1 = c * min / 1234567890101112;
	double res2 = res1 * 1234567890101112/min;
	std::cout << res1 << std::endl << res2 << std::endl;
}

Eigen::Vector2d zerosquadpol(double alpha, double beta) {
	Eigen::Vector2d z;
	double D = std::pow(alpha, 2) - 4 * beta;
	if (D < 0) throw "no real zeros";
	else {
		double wD = std::sqrt(D);
		z << (-alpha - wD) / 2, (-alpha + wD) / 2;
	}
	return z;
}

void compzeros() {
	int n = 100;
	Eigen::MatrixXd res(n, 4);
	Eigen::VectorXd gamma = Eigen::VectorXd::LinSpaced(n, 2, 992);
	for (int i = 0; i < n; ++i) {
		double alpha = -(gamma(i) + 1. / gamma(i));
		double beta = 1.;
		Eigen::Vector2d z1 = zerosquadpol(alpha, beta);
		Eigen::Vector2d z1 = zerosquadpolstab(alpha, beta);
	}

	
}

void diffq() {
	double h = 0.1, x = 0.0;
	for (int i = 1; i <= 16; ++i) {
		double df = (exp(x + h) - exp(x)) / h;
		std::cout << std::setprecision(14) << std::fixed;
		std::cout << std::setw(5) << -i << std::setw(20) << df - 1 << std::endl;
		h /= 10;
	}
}

void numericaldifferentiation() {
	/*typedef std::mpreal numeric_t;
	int n = 7, k = 13;
	Eigen::VectorXi bits(n); bits << 10, 30, 50, 70, 90, 110, 130;
	Eigen::MatrixXd experr(13, bits.size()+1);
	for (int i = 0; i < n; ++i)
	{
		numeric_t::set_default_prec(bits(i));
		numeric_t x = "1.1";
		for (int j = 0; j < k;  ++j) {
			numeric_t h = std::pow("2", -1 - 5 * j);
			experr(j, i + 1) = std::abs((mpfr::exp(x + h) - mpfr::exp(x)) / h - mpfr::exp(x)).toDouble();
			experr(j, 0) = h.toDouble();
		}
	}*/
}

Eigen::VectorXd zerosquadpolstab(double alpha, double beta) {
	Eigen::Vector2d z(2);
	double D = std::pow(alpha, 2) - 4 * beta;

	if (D < 0) throw "no real zeros";
	else {
		double wD = std::sqrt(D);
		 
		if (alpha >= 0) {
			double t = 0.5*(-alpha - wD);
			z << t, beta / t;
		}
		else {
			double t = 0.5*(-alpha + wD);
			z << beta / t, t;
			
		}
	}
	return z;

}

Eigen::MatrixXd ApproxPlinstable(double tol = 1e-8, int maxlt = 50) {
	double s = std::sqrt(3) / 2.; double An = 3.*s;
	unsigned int n = 6, it = 0;
	Eigen::MatrixXd res(maxlt, 4);
	res(it, 0) = n; res(it, 1) = An;
	//res(it, 2) = An - M_PI; res(it, 3) = 3;
	while (it < maxlt && s > tol) {
		s = std::sqrt((1.- std::sqrt(1.-s*s))/2.);
		n *= 2; An = n / 3.*s;
		++it;
		res(it, 0) = n; res(it, 1) = An;
		//res(it, 2) = An - M_PI; res(it, 3)=s;

	}
	return res.topRows(it);
}

Eigen::MatrixXd apprpistable(double tol = 1e-8, int maxlt = 50) {
	double s = std::sqrt(3) / 2.; double An = 3.*s;
	unsigned int n = 6, it = 0;
	Eigen::MatrixXd res(maxlt, 4);
	res(it, 0) = n; res(it, 1) = An;
	//res(it, 2) = An - M_PI; res(it, 3) = 3;
	while (it < maxlt && s > tol) {
		s = s/ std::sqrt(2*(1+ std::sqrt((1+s)*(1-s))));
		n *= 2; An = n / 2.*s;
		++it;
		res(it, 0) = n; res(it, 1) = An;
		//res(it, 2) = An - M_PI; res(it, 3)=s;

	}
	return res.topRows(it);
}

double expeval(double x, double tol=1e-8) {
	double y = 1.0, term = 1.0;
	long int k = 1;
	while (abs (term) > tol*y) {
		term *= x / k;
		y += term;
		++k;
	}
	return y;
}

int main()
{
	
	// eigenTypeDemo<double>((2)); 
	//matArray(3, 3);
	//storageOrder();


	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
