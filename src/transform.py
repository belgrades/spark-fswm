def main():
    sequences, word = [], ""
    with open("bacteria.fasta", "r") as file:
        with open("bacteria2.fasta", "a") as output:
            for idx, line in enumerate(file.readlines()):
                if line[0] == ">":
                    if word is not "":
                        output.write(word + "\n")
                        word = ""
                    output.write(line)
                    # New sequence
                else:
                    # New sequence
                    word += line.replace("\n", "")
            output.write(line + "\n")


if __name__ == "__main__":
    main()
