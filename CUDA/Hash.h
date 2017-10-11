#pragma once
#include <string>
#include <iostream>
#ifndef HASH
#define HASH

namespace hash 
{
	class Hash
	{
	private:

		std::string hash;
		int RecievingExistCodes(int);
		int GetControlSum(std::string);

	public:

		Hash();
		~Hash();
		std::string GetHash(std::string, unsigned int);
		void Clear();
	
	};
}
#endif HASH