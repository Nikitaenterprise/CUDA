#include "Hash.h"

int hash::Hash::RecievingExistCodes(int x)
{
	x += 256;
	//std::cout << "\t\t#########################" << std::endl;
	//std::cout << "\t\tBEGIN RECIEVINGEXISTCODE" << std::endl;
	//std::cout << "\t\tx = " << x << std::endl;
	//std::cout << "\t\tBEGIN cycle" << std::endl;
	while (!(((x <= 57) && (x >= 48)) || ((x <= 90) && (x >= 65)) || ((x <= 122) && (x >= 97))))
	{
		if (x < 48) x += 24;
		else x -= 47;
		//std::cout << "\t\tx = " << x << std::endl;
	}
	//std::cout << "\t\tEND cycle" << std::endl;
	//std::cout << "\t\tEND RECIEVINGEXISTCODE" << std::endl;
	//std::cout << "\t\t#########################" << std::endl;
	return x;
}

int hash::Hash::GetControlSum(std::string str)
{
	unsigned int sault = 0, strlen = 0;
	//std::cout << "\t#########################" << std::endl;
	//std::cout << "\tBEGIN GETCONTROLSUM" << std::endl;
	//std::cout << "\tstr = " << str << std::endl;
	//std::cout << "\tBEGIN cycle" << std::endl;
	for (; strlen < str.size(); strlen++)
	{
		sault += int(str[strlen]);
		//std::cout << "\t\tstr[strlen] = " << str[strlen] << std::endl;
		//std::cout << "\t\tint(str[strlen]) = " << int(str[strlen]) << std::endl;
		//std::cout << "\t\tsault = " << sault << std::endl;
	}
	//std::cout << "\tEND cycle" << std::endl;
	//std::cout << "\tsault = " << sault << std::endl;
	//std::cout << "\tEND GETCONTROLSUM" << std::endl;
	//std::cout << "\t#########################" << std::endl;
	return sault;
}

hash::Hash::Hash()
{
}

hash::Hash::~Hash()
{
}

std::string hash::Hash::GetHash(std::string userString, unsigned int lengthHash)
{
	if (lengthHash > 3)
	{
		unsigned int minLen = 2;
		unsigned int realMinLen = 0;
		unsigned int originalSault = this->GetControlSum(userString);
		unsigned int originalLengthStr = userString.size();

		//std::cout << "#########################" << std::endl;
		//std::cout << "minLen = " << minLen << std::endl;
		//std::cout << "realMinLen = " << realMinLen << std::endl;
		//std::cout << "originalSault = " << originalSault << std::endl;
		//std::cout << "originalLengthStr = " << originalLengthStr << std::endl;
		//std::cout << "hash = " << hash << std::endl;
		//std::cout << "userString = " << userString << std::endl;
		//std::cout << "lengthHash = " << lengthHash << std::endl;

		//std::cout << "BEGIN cycle" << std::endl;
		while (minLen <= lengthHash)
		{
			realMinLen = (minLen *= 2);
			//std::cout << "\tminLen = " << minLen << std::endl;
			//std::cout << "\trealMinLen = " << realMinLen << std::endl;
		}
		//std::cout << "END cycle" << std::endl;
		//std::cout << "BEGIN cycle" << std::endl;
		while (minLen < originalLengthStr) 
		{
			minLen *= 2;
			//std::cout << "\tminLen = " << minLen << std::endl;
		}
		//std::cout << "END cycle" << std::endl;

		if ((minLen - originalLengthStr) < minLen) minLen *= 2;

		int addCount = minLen - originalLengthStr;
		//std::cout << "minLen - originalLengthStr = " << minLen << " - " << originalLengthStr << std::endl;
		//std::cout << "addCount = " << addCount << std::endl;
		//std::cout << "BEGIN cycle" << std::endl;
		for (int i = 0; i < addCount; i++)
		{
			userString += this->RecievingExistCodes(userString[i] + userString[i + 1]);
			//std::cout << "\tuserString[i] = " << userString[i] << std::endl;
			//std::cout << "\tuserString[i + 1] = " << userString[i + 1] << std::endl;
			//std::cout << "\tthis->RecievingExistCodes(userString[i] + userString[i + 1]) = " << this->RecievingExistCodes(userString[i] + userString[i + 1]) << std::endl;
			//std::cout << "\tuserString = " << userString << std::endl;
		}
		//std::cout << "END cycle" << std::endl;
		int maxSault = this->GetControlSum(userString);
		//std::cout << "maxSault = " << maxSault << std::endl;
		int maxLengthStr = userString.size();
		//std::cout << "maxLengthStr = " << maxLengthStr << std::endl;
		//std::cout << "minLen = " << minLen << std::endl;
		//std::cout << "realMinLen = " << realMinLen << std::endl;
		//std::cout << "originalSault = " << originalSault << std::endl;
		//std::cout << "originalLengthStr = " << originalLengthStr << std::endl;
		//std::cout << "hash = " << hash << std::endl;
		//std::cout << "userString = " << userString << std::endl;
		//std::cout << "lengthHash = " << lengthHash << std::endl;
		//std::cout << "\tBEGIN cycle" << std::endl;
		while (userString.size() != realMinLen)
		{
			for (int i = 0, center = userString.size() / 2; i < center; i++) 
				this->hash += this->RecievingExistCodes(userString[center - i] + userString[center + i]);

			userString = this->hash;
			this->hash.clear();
			//std::cout << "\tuserString = " << userString << std::endl;
		}
		//std::cout << "\tEND cycle" << std::endl;
		//std::cout << "userString = " << userString << std::endl;
		unsigned int rem = realMinLen - lengthHash;
		//std::cout << "realMinLen - lengthHash = " << realMinLen << " - " << lengthHash << std::endl;
		//std::cout << "rem = " << rem << std::endl;
		//std::cout << "\tBEGIN cycle" << std::endl;
		for (unsigned int i = 0, countCompress = realMinLen / rem; this->hash.size() < (lengthHash - 4); i++)
		{
			if (i % countCompress == 0) this->hash += this->RecievingExistCodes(userString[i] + userString[++i]);
			else this->hash += userString[i];
			//std::cout << "\tcountCompress = " << countCompress << std::endl;
			//std::cout << "\thash = " << hash << std::endl;
		}
		//std::cout << "\tEND cycle" << std::endl;
		this->hash += this->RecievingExistCodes(originalSault);
		//std::cout << "hash = " << hash << std::endl;
		//std::cout << "originalSault = " << originalSault << std::endl;
		this->hash += this->RecievingExistCodes(originalLengthStr);
		//std::cout << "hash = " << hash << std::endl;
		//std::cout << "originalLengthStr = " << originalLengthStr << std::endl;
		this->hash += this->RecievingExistCodes(maxSault);
		//std::cout << "hash = " << hash << std::endl;
		//std::cout << "maxSault = " << maxSault << std::endl;
		this->hash += this->RecievingExistCodes(maxLengthStr);
		//std::cout << "hash = " << hash << std::endl;
		//std::cout << "maxLengthStr = " << maxLengthStr << std::endl;
		//std::cout << "#########################" << std::endl;
		return this->hash;
	}
	return "";
}

void hash::Hash::Clear()
{
	hash.clear();
}
