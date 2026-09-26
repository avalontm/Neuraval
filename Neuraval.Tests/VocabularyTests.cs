using Neuraval.Core.Tokenizers;
using Xunit;

namespace Neuraval.Tests
{
    public class VocabularyTests
    {
        [Fact]
        public void AddToken_AssignsSequentialIds()
        {
            var vocabulary = new Vocabulary();

            int firstId = vocabulary.AddToken("a");
            int secondId = vocabulary.AddToken("b");

            Assert.Equal(0, firstId);
            Assert.Equal(1, secondId);
        }

        [Fact]
        public void AddToken_RepeatedToken_ReturnsSameId()
        {
            var vocabulary = new Vocabulary();

            int firstId = vocabulary.AddToken("a");
            int repeatedId = vocabulary.AddToken("a");

            Assert.Equal(firstId, repeatedId);
            Assert.Equal(1, vocabulary.Count);
        }

        [Fact]
        public void GetId_UnknownToken_ReturnsFallback()
        {
            var vocabulary = new Vocabulary();
            vocabulary.AddToken("a");

            Assert.Equal(-1, vocabulary.GetId("missing", unknownId: -1));
        }

        [Fact]
        public void GetToken_UnknownId_ReturnsFallback()
        {
            var vocabulary = new Vocabulary();
            vocabulary.AddToken("a");

            Assert.Equal("<unk>", vocabulary.GetToken(99, "<unk>"));
        }

        [Fact]
        public void SaveState_LoadState_PreservesTokenIds()
        {
            var vocabulary = new Vocabulary();
            vocabulary.AddToken("a");
            vocabulary.AddToken("b");
            vocabulary.AddToken("c");

            var reloaded = Vocabulary.LoadState(vocabulary.SaveState());

            Assert.Equal(vocabulary.Count, reloaded.Count);
            Assert.True(reloaded.TryGetId("b", out var id));
            Assert.Equal(1, id);
        }
    }
}
