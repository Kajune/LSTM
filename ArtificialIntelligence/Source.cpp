#include "LSTM.hpp"

int main() {
	LSTM::Node<2, 3> n1;

	n1.Forward({ 1, 2, 3 });
	;

	return 0;
}
