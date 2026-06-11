#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;

// SVD para PCA - redução de dimensionalidade
MatrixXd pca_reduce(const MatrixXd& X, int k) {
    // Centralizando os dados: X_c = X - mean()
    MatrixXd X_c = X.rowwise() - X.colwise().mean();

    // SVD via Jacobi (mais estável numericamente)
    BDCSVD<MatrixXd> svd(X_c, ComputeThinU | ComputeThinV);

    // Componentes principais: k primeiras colunas de V
    MatrixXd V_k = svd.matrixV().leftCols(k);

    // Projetando: X_reduzido = X_c x V_k
    return X_c * V_k;
}

// Regressão Linear via equações normais: θ = (XᵀX)⁻¹Xᵀy
VectorXd linear_regression(const MatrixXd& X, const VectorXd& y) {
    // colPivHouseholderQr()
    return X.colPivHouseholderQr().solve(y);
}

int main() {
    // Conjunto de dados fictício: 100.000 amostras com 50 atributos
    MatrixXd X = MatrixXd::Random(100000, 50);
    VectorXd y = VectorXd::Random(100000);

    // PCA para 2D
    MatrixXd X_2d = pca_reduce(X, 2);
    std::cout << "Shape após PCA: "
              << X_2d.rows() << "x" << X_2d.cols() << "\n";

    // Regressão
    VectorXd theta = linear_regression(X, y);
    
    // squaredNorm()
    double mse = (X * theta - y).squaredNorm() / y.size();
    std::cout << "MSE: " << mse << "\n";

    return 0;
}
