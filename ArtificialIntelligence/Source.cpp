#include "LSTM.hpp"

int main() {
	LSTM::Node<2, 2> n1, n2;

	n2.Forward(n1.Forward({ 1, 2, 3 }));

	return 0;
}
